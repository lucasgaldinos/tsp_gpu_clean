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
 *   - Fujimoto & Tsutsui (2011). "A Highly-Parallel TSP Solver for a GPU Computing Platform"
 *   - Croes (1958). "A Method for Solving Traveling Salesman Problems"

 * Author: GPU Debugging Session
 * Date: 2025-01-27
 * Updated: 2025-11-14 (Phase 3.5 - Kernel extraction)
 * >[!warning]
 * >- This references are out of place, please double check. Is this really the same method as fujimoto or is it based on tsplogo/rocki and suda.
 * >- change the variables to descriptive names
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
        int node_i = s_tour[i];
        int node_i1 = s_tour[i + 1];
        
        // Serial inner loop over j
        for (int j = i + 2; j < n; ++j) {
            int node_j = s_tour[j];
            int node_j1 = s_tour[(j + 1) % n];
            
            // Calculate delta (new - old: negative = improvement)
            int old_dist = dist[node_i * n + node_i1] + 
                           dist[node_j * n + node_j1];
            int new_dist = dist[node_i * n + node_j] + 
                           dist[node_i1 * n + node_j1];
            int delta = new_dist - old_dist;  // Negative = improvement
            
            // Update best for this thread (delta < 0 means improvement)
            if (delta < my_delta) {
                my_delta = delta;
                my_swap_i = i;
                my_swap_j = j;
            }
        }
    }
    
    // Store thread's best in shared memory for reduction
    s_deltas[tid] = my_delta;  // int delta
    s_swap_i[tid] = my_swap_i;
    s_swap_j[tid] = my_swap_j;
    __syncthreads();
    
    // Parallel reduction to find minimum delta (most negative = best improvement)
    // For non-power-of-2 block sizes, we need special handling
    // Strategy: Parallel reduce as much as possible, then serial reduction for remainder
    
    // Parallel reduction for power-of-2 portion
    for (unsigned int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            if (tid + s < block_size && s_deltas[tid + s] < s_deltas[tid]) {
                s_deltas[tid] = s_deltas[tid + s];
                s_swap_i[tid] = s_swap_i[tid + s];
                s_swap_j[tid] = s_swap_j[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Final reduction: use thread 0 to serially reduce remaining values
    // This guarantees correctness for any block_size (power-of-2 or not)
    if (tid == 0) {
        for (int i = 1; i < block_size; i++) {
            if (s_deltas[i] < s_deltas[0]) {
                s_deltas[0] = s_deltas[i];
                s_swap_i[0] = s_swap_i[i];
                s_swap_j[0] = s_swap_j[i];
            }
        }
    }
    __syncthreads();
    
    // Apply best swap (Thread 0 only)
    if (tid == 0 && s_deltas[0] < 0) {
        int swap_i = s_swap_i[0];
        int swap_j = s_swap_j[0];
        
        // Reverse segment [i+1, j]
        int left = swap_i + 1;
        int right = swap_j;
        while (left < right) {
            int temp = s_tour[left];
            s_tour[left] = s_tour[right];
            s_tour[right] = temp;
            left++;
            right--;
        }
    }
    __syncthreads();
    
    // FIX: Write back using loop to handle n > block_size
    for (int i = tid; i < n; i += block_size) {
        tour[i] = s_tour[i];
    }
}
