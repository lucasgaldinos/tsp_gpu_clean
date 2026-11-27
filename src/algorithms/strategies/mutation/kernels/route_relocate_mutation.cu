/**
 * GPU 0-Delimited VRP Route Relocate Mutation Kernel (`DP17-7`, `BUG-114`)
 *
 * Unbiased intra/inter-route customer relocate allowing insertion at ANY slot
 * `pos2 in [0, width) \ {pos1}` — including index 0 (start of a route) and empty
 * delimiter `0` vehicle slots (opening new routes when unused `0`s exist).
 * Preserves exact customer bijection `{1..dimension-1}` and depot `0` count with zero D2H.
 */

extern "C" __global__ void route_relocate_mutation_kernel(
    const int *__restrict__ routes,           // [batch_size, width]
    int *__restrict__ out_routes,             // [batch_size, width]
    const unsigned char *__restrict__ active, // [batch_size]
    const int *__restrict__ k1_arr,           // [batch_size] or nullptr
    const double *__restrict__ u1_arr,        // [batch_size] fallback when k1_arr == nullptr
    const int *__restrict__ s2_arr,           // [batch_size] in [0, width - 1)
    const int batch_size,
    const int width)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
    {
        return;
    }

    const int *src = routes + (size_t)idx * (size_t)width;
    int *dst = out_routes + (size_t)idx * (size_t)width;

    int n_cust = 0;
    for (int col = 0; col < width; ++col)
    {
        const int val = src[col];
        dst[col] = val;
        if (val > 0)
        {
            n_cust += 1;
        }
    }

    if (active[idx] == 0 || n_cust < 1 || width < 2)
    {
        return;
    }

    int k1 = 0;
    if (k1_arr != nullptr)
    {
        k1 = k1_arr[idx];
    }
    else
    {
        k1 = (int)(u1_arr[idx] * (double)n_cust);
        if (k1 >= n_cust)
        {
            k1 = n_cust - 1;
        }
    }

    if (k1 < 0 || k1 >= n_cust)
    {
        return;
    }

    int pos1 = -1;
    int seen = 0;
    for (int col = 0; col < width; ++col)
    {
        if (dst[col] > 0)
        {
            if (seen == k1)
            {
                pos1 = col;
                break;
            }
            seen += 1;
        }
    }

    if (pos1 < 0)
    {
        return;
    }

    const int s2 = s2_arr[idx];
    const int pos2 = s2 + ((s2 >= pos1) ? 1 : 0);
    if (pos2 < 0 || pos2 >= width || pos2 == pos1)
    {
        return;
    }

    const int cust = dst[pos1];
    if (pos1 < pos2)
    {
        for (int c = pos1; c < pos2; ++c)
        {
            dst[c] = dst[c + 1];
        }
        dst[pos2] = cust;
    }
    else
    {
        for (int c = pos1; c > pos2; --c)
        {
            dst[c] = dst[c - 1];
        }
        dst[pos2] = cust;
    }
}
