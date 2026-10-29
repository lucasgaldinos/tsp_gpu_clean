/**
 * GPU 2-opt Batch Processing Kernel
 * 
 * Performs one pass of 2-opt improvement for multiple TSP tours in parallel.
 * Each CUDA block processes one tour independently.
 * 
 * Algorithm:
 *   1. Each block loads its tour into shared memory
 *   2. Parallel evaluation of all (i,j) swap pairs within block
 *   3. Block-level reduction finds best improvement
 *   4. Apply best swap in-place
 *   5. Write back improved tour
 * 
 * Performance Notes:
 *   - No inter-block synchronization (tours are independent)
 *   - Shared memory per block reduces global memory traffic
 *   - Batch processing eliminates per-tour kernel launch overhead
 *   - Expected 5-10x speedup for population-based algorithms
 * 
 * Usage Pattern (Phase 3 Batch API):
 *   - Genetic Algorithm: Improve entire population in one kernel launch
 *   - Multistart: Process multiple independent runs in parallel
 *   - Grid: num_tours blocks, each with up to 256 threads
 * 
 * References:
 *   - Fujimoto & Tsutsui (2011). "A Highly-Parallel TSP Solver for a GPU Computing Platform"
 *   - Phase 3 Implementation: Batch API for metaheuristics
 * 
 * Author: GPU Debugging Session
 * Date: 2025-01-27
 * Updated: 2025-11-14 (Phase 3.5 - Kernel extraction, ASCII-only)
* >[!warning]
 * >- This references are out of place, please double check. Is this really the same method as fujimoto or is it based on tsplogo/rocki and suda.
 * >- change the variables to descriptive names
 */

extern "C" __global__
void two_opt_batch_kernel(
    int *tours,          // Flattened: num_tours * n
    const int *dist,    // n * n distance matrix (int32, TSPLIB compliant)
    int n,               // Number of nodes per tour
    int num_tours        // Number of tours
) {
    extern __shared__ int shared_mem[];
    
    // Partition shared memory (per-block)
    int *s_deltas = shared_mem;
    int *s_swap_i = (int*)&s_deltas[blockDim.x];
    int *s_swap_j = (int*)&s_swap_i[blockDim.x];
    int *s_tour = (int*)&s_swap_j[blockDim.x];
    
    int tour_idx = blockIdx.x;  // Which tour this block processes
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Offset to this tour's data
    int *tour = tours + (tour_idx * n);
    
    // Load tour into shared memory
    for (int i = tid; i < n; i += block_size) {
        s_tour[i] = tour[i];
    }
    __syncthreads();
    
    // Initialize shared memory for this thread
    s_deltas[tid] = 0;
    s_swap_i[tid] = -1;
    s_swap_j[tid] = -1;
    
    // Each thread evaluates a subset of i positions
    for (int i = tid; i < n - 2; i += block_size) {
        int node_i = s_tour[i];
        int node_i1 = s_tour[i + 1];
        
        // Serial inner loop over j
        for (int j = i + 2; j < n; ++j) {
            int node_j = s_tour[j];
            int node_j1 = s_tour[(j + 1) % n];
            
            // Calculate delta
            int old_dist = dist[node_i * n + node_i1] + 
                           dist[node_j * n + node_j1];
            int new_dist = dist[node_i * n + node_j] + 
                           dist[node_i1 * n + node_j1];
            int delta = new_dist - old_dist;
            
            // Update best for this thread
            if (delta < s_deltas[tid]) {
                s_deltas[tid] = delta;
                s_swap_i[tid] = i;
                s_swap_j[tid] = j;
            }
        }
    }
    __syncthreads();
    
    // Parallel reduction to find minimum delta across all threads in this block
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
    
    // Write back to global memory
    for (int i = tid; i < n; i += block_size) {
        tour[i] = s_tour[i];
    }
}
