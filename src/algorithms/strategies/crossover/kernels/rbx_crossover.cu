/**
 * GPU 0-Delimited VRP Route-Based Crossover (RBX, Potvin & Bengio 1996) Kernel (`DP17-2`, `DP17-3`, `BUG-114`)
 *
 * Evaluates `(batch_size, width)` `int32` 0-delimited routes in parallel (one thread per child).
 * Copies one selected non-empty route from parent 1, appends remaining unvisited customers in
 * parent 2's relative order splitting on BOTH capacity (Q) and max route distance (L_max,
 * honoring `return_to_depot` via `gpu_edge_distance`), never overflowing `width`, and pads
 * trailing slots with `0`. Zero D2H transfers.
 */

#include "pairwise_device.cuh"

extern "C" __global__ void rbx_crossover_kernel(
    const int *__restrict__ parents,         // [batch_size, width]
    int *__restrict__ offspring,             // [batch_size, width]
    const int *__restrict__ demands,         // [dimension]
    const int *__restrict__ distances,       // [dimension, dimension] or nullptr
    const double *__restrict__ coords,       // [dimension, 2|3] or nullptr
    const int *__restrict__ route_picks,     // [batch_size] non-negative ints
    unsigned char *__restrict__ visited_buf, // [batch_size, dimension]
    const int batch_size,
    const int width,
    const int dimension,
    const int edge_type_code,
    const int return_to_depot,
    const int has_capacity,
    const long long capacity,
    const int has_max_distance,
    const double max_route_distance)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
    {
        return;
    }

    const bool use_matrix = (distances != nullptr);
    const int *p1 = parents + (size_t)idx * (size_t)width;
    const int p2_idx = (idx + 1 < batch_size) ? (idx + 1) : 0;
    const int *p2 = parents + (size_t)p2_idx * (size_t)width;
    int *child = offspring + (size_t)idx * (size_t)width;
    unsigned char *visited = visited_buf + (size_t)idx * (size_t)dimension;

    for (int i = 0; i < dimension; ++i)
    {
        visited[i] = 0;
    }

    // 1. Count non-empty routes in parent 1
    int n_routes = 0;
    bool in_route = false;
    for (int col = 0; col < width; ++col)
    {
        const int node = p1[col];
        if (node > 0)
        {
            if (!in_route)
            {
                in_route = true;
                n_routes += 1;
            }
        }
        else
        {
            in_route = false;
        }
    }

    int write_pos = 0;
    int copied_count = 0;
    int used_routes = 0;
    const int k_max = (width > dimension) ? (width - dimension) : 1;

    long long p1_load = 0LL;
    long long p1_dist = 0LL;
    int p1_last = 0;

    // 2. Copy the selected route from parent 1
    if (n_routes > 0)
    {
        const int target_route = route_picks[idx] % n_routes;
        int cur_route = -1;
        in_route = false;
        if (write_pos < width)
        {
            child[write_pos++] = 0;
        }
        for (int col = 0; col < width; ++col)
        {
            const int node = p1[col];
            if (node > 0)
            {
                if (!in_route)
                {
                    in_route = true;
                    cur_route += 1;
                }
                if (cur_route == target_route && node < dimension && visited[node] == 0)
                {
                    if (write_pos < width)
                    {
                        child[write_pos++] = node;
                    }
                    visited[node] = 1;
                    copied_count += 1;
                    p1_load += (long long)demands[node];
                    if (has_max_distance != 0)
                    {
                        p1_dist += (long long)gpu_edge_distance(
                            p1_last, node, distances, coords, dimension, edge_type_code, use_matrix);
                    }
                    p1_last = node;
                }
            }
            else
            {
                in_route = false;
            }
        }
        if (copied_count > 0)
        {
            used_routes = 1;
            if (write_pos < width)
            {
                child[write_pos++] = 0;
            }
        }
        else
        {
            write_pos = 0;
        }
    }

    // 3. Append remaining unvisited customers in parent 2's relative order
    int rem_unvisited = (dimension - 1) - copied_count;
    bool cur_open = false;
    long long cur_load = 0LL;
    long long cur_dist = 0LL;
    int last_node = 0;

    for (int col = 0; col < width && rem_unvisited > 0; ++col)
    {
        const int node = p2[col];
        if (node <= 0 || node >= dimension || visited[node] != 0)
        {
            continue;
        }

        const long long q = (long long)demands[node];
        if (!cur_open)
        {
            if (used_routes == 0)
            {
                if (write_pos + 1 + rem_unvisited <= width)
                {
                    child[write_pos++] = 0;
                }
                used_routes = 1;
                cur_open = true;
                cur_load = q;
                cur_dist = (has_max_distance != 0)
                               ? (long long)gpu_edge_distance(
                                     0, node, distances, coords, dimension, edge_type_code, use_matrix)
                               : 0LL;
                last_node = node;
            }
            else if (used_routes < k_max && (write_pos + rem_unvisited < width))
            {
                // Previous route already wrote its trailing 0 delimiter
                used_routes += 1;
                cur_open = true;
                cur_load = q;
                cur_dist = (has_max_distance != 0)
                               ? (long long)gpu_edge_distance(
                                     0, node, distances, coords, dimension, edge_type_code, use_matrix)
                               : 0LL;
                last_node = node;
            }
            else
            {
                // Merge into last route (Parent 1's copied route) by reclaiming its trailing 0
                if (write_pos > 1 && child[write_pos - 1] == 0)
                {
                    write_pos -= 1;
                }
                cur_open = true;
                cur_load = p1_load + q;
                cur_dist = (has_max_distance != 0)
                               ? p1_dist + (long long)gpu_edge_distance(
                                               p1_last, node, distances, coords, dimension, edge_type_code, use_matrix)
                               : 0LL;
                last_node = node;
            }
        }
        else
        {
            const long long step_dist = (has_max_distance != 0)
                                            ? (long long)gpu_edge_distance(
                                                  last_node, node, distances, coords, dimension, edge_type_code, use_matrix)
                                            : 0LL;
            const long long ret_dist = (has_max_distance != 0 && return_to_depot != 0)
                                           ? (long long)gpu_edge_distance(
                                                 node, 0, distances, coords, dimension, edge_type_code, use_matrix)
                                           : 0LL;
            const bool cap_viol = (has_capacity != 0) && (cur_load + q > capacity);
            const bool dist_viol = (has_max_distance != 0) &&
                                   ((double)(cur_dist + step_dist + ret_dist) > max_route_distance);

            if ((cap_viol || dist_viol) &&
                (used_routes < k_max) &&
                (write_pos + 1 + rem_unvisited < width))
            {
                child[write_pos++] = 0;
                used_routes += 1;
                cur_load = q;
                cur_dist = (has_max_distance != 0)
                               ? (long long)gpu_edge_distance(
                                     0, node, distances, coords, dimension, edge_type_code, use_matrix)
                               : 0LL;
                last_node = node;
            }
            else
            {
                cur_load += q;
                cur_dist += step_dist;
                last_node = node;
            }
        }

        if (write_pos < width)
        {
            child[write_pos++] = node;
        }
        visited[node] = 1;
        rem_unvisited -= 1;
    }

    // 4. Pad all remaining slots with 0 delimiters
    while (write_pos < width)
    {
        child[write_pos++] = 0;
    }
}
