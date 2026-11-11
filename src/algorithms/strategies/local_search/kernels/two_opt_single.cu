/**
 * GPU 2-opt Single Tour Kernel
 * 
 * Performs one pass of 2-opt improvement for a single TSP tour.
 * Uses shared memory for efficient parallel evaluation of all possible swaps.
 * 
 * Algorithm:
 *   1. Load tour into shared memory
 *   2. Each thread evaluates subset of (i,j) swap pairs
 *   3. Parallel reduction finds best improvement (minimum delta)
 *   4. Apply best swap in-place
 *   5. Write back improved tour
 * 
 * Performance Notes:
 *   - Shared memory reduces global memory accesses
 *   - Parallel reduction minimizes synchronization
 *   - In-place reversal avoids temporary allocations
 * 
 * References:
 *   - Croes, G. A. (1958). "A Method for Solving Traveling-Salesman Problems."
 *     Operations Research, 6(6), 791-812.
 *   - Fujimoto, N., & Tsutsui, S. (2011). "A Highly-Parallel TSP Solver for a
 *     GPU Computing Platform." ICNAAM, Vol. 1389, pp. 264-267. (Shared-memory
 *     tour representation per CUDA thread block).
 *   - Rocki, K., & Suda, R. (2012). "Accelerating 2-opt and 3-opt Local Search
 *     Using GPU in the Travelling Salesman Problem." IEEE HPCS, pp. 489-495.
 *     (Cooperative parallel evaluation of (i, j) pairs and block reduction).
 */

extern "C" __global__
void two_opt_kernel(
    int *tour,
    const int *dist,
    int n,
    int tour_idx
) {
    extern __shared__ int shared_mem[];
    
    // Partition shared memory
    int *s_deltas = shared_mem;
    int *s_swap_i = (int*)&s_deltas[blockDim.x];
    int *s_swap_j = (int*)&s_swap_i[blockDim.x];
    int *s_tour = (int*)&s_swap_j[blockDim.x];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // FIX: Load tour using loop to handle n > block_size
    for (int i = tid; i < n; i += block_size) {
        s_tour[i] = tour[i];
    }
    __syncthreads();
    
    // Initialize this thread's best delta
    int my_delta = 0;
    int my_swap_i = -1;
    int my_swap_j = -1;
    
    // Each thread evaluates a subset of i positions
    for (int i = tid; i < n - 2; i += block_size) {
        // Hoisted row endpoints and current edge cost (`Task 10.16`)
        int node_i = s_tour[i];
        int node_i1 = s_tour[i + 1];
        const int *row_i = dist + (size_t)node_i * (size_t)n;
        const int *row_i1 = dist + (size_t)node_i1 * (size_t)n;
        int d_ab = row_i[node_i1];
        // Carry-over: `node_j` of iteration `j + 1` is `node_j1` of iteration `j`
        int node_j = s_tour[i + 2];

        // Serial inner loop over j (upper triangle only)
        for (int j = i + 2; j < n; ++j) {
            int node_j1 = s_tour[(j + 1) % n];

            // Calculate delta (new - old: negative = improvement)
            int old_dist = d_ab + dist[(size_t)node_j * (size_t)n + (size_t)node_j1];
            int new_dist = row_i[node_j] + row_i1[node_j1];
            int delta = new_dist - old_dist;  // Negative = improvement

            // Update best for this thread (delta < 0 means improvement)
            if (delta < my_delta) {
                my_delta = delta;
                my_swap_i = i;
                my_swap_j = j;
            }
            node_j = node_j1;
        }
    }
    
    // Store thread's best in shared memory for reduction
    s_deltas[tid] = my_delta;  // int delta
    s_swap_i[tid] = my_swap_i;
    s_swap_j[tid] = my_swap_j;
    __syncthreads();
    
    // Block reduction to the exact minimum delta.
    //
    // Phase 1 folds thread partials in parallel while the stride still spans more than
    // one warp. Phase 2 lets thread 0 fold only the surviving `min(block_size, 64)`
    // partials -- the pre-fix code rescanned all `block_size` of them after the tree had
    // already merged the upper half (`BUG-29`).
    for (unsigned int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s && tid + s < block_size && s_deltas[tid + s] < s_deltas[tid]) {
            s_deltas[tid] = s_deltas[tid + s];
            s_swap_i[tid] = s_swap_i[tid + s];
            s_swap_j[tid] = s_swap_j[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int remaining = block_size < 64 ? block_size : 64;
        for (int i = 1; i < remaining; i++) {
            if (s_deltas[i] < s_deltas[0]) {
                s_deltas[0] = s_deltas[i];
                s_swap_i[0] = s_swap_i[i];
                s_swap_j[0] = s_swap_j[i];
            }
        }
    }
    __syncthreads();

    // `s_deltas[0]` is now the exact minimum, so a pass with no improving move skips
    // the reversal branch instead of synchronizing for nothing (`BUG-29`).
    if (s_deltas[0] < 0) {
        int swap_i = s_swap_i[0];
        int swap_j = s_swap_j[0];
        
        // Reverse segment [swap_i + 1, swap_j]
        int left = swap_i + 1;
        int right = swap_j;
        int num_swaps = (right - left + 1) / 2;
        for (int k = tid; k < num_swaps; k += block_size) {
            int temp = s_tour[left + k];
            s_tour[left + k] = s_tour[right - k];
            s_tour[right - k] = temp;
        }
    }
    __syncthreads();
    
    // FIX: Write back using loop to handle n > block_size
    for (int i = tid; i < n; i += block_size) {
        tour[i] = s_tour[i];
    }
}
