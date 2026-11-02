/*
 * GPU-native Genetic Algorithm Kernel for TSP with Early Stopping
 * Based on the concepts from Fujimoto & Tsutsui (2011).
 * 
 * Extended with early stopping capability:
 *   - Optimal cost detection (stops when optimal reached)
 *   - Stagnation detection (stops when no improvement for N generations)
 *
 * This kernel is intended to be a simplified proof-of-concept and does not
 * implement the full complexity of the paper's proposed memory optimizations
 * or advanced crossover/mutation operators. It serves as a structural
 * template for a fully GPU-resident GA.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define MAX_CITIES 2048 // Max cities for stack-based arrays

// --- Utility Functions ---

// Device function to calculate the total distance of a tour
__device__ long long calculate_fitness(int* tour, int num_cities, int* distances) {
    long long total_distance = 0;
    for (int i = 0; i < num_cities - 1; ++i) {
        total_distance += distances[tour[i] * num_cities + tour[i+1]];
    }
    total_distance += distances[tour[num_cities - 1] * num_cities + tour[0]];
    return total_distance;
}

// Device function to reverse a segment of a tour
__device__ void reverse_segment(int* tour, int start, int end) {
    while (start < end) {
        int temp = tour[start];
        tour[start] = tour[end];
        tour[end] = temp;
        start++;
        end--;
    }
}

// Device function to perform a 2-Opt local search on a single tour
__device__ void two_opt_device(int* tour, int num_cities, int* distances, int max_iterations) {
    bool improved = true;
    int iterations = 0;
    while (improved && iterations < max_iterations) {
        improved = false;
        iterations++;
        for (int i = 0; i < num_cities - 1; i++) {
            for (int j = i + 1; j < num_cities; j++) {
                int i_next = (i + 1);
                int j_next = (j + 1) % num_cities;

                // Current distance of the two edges (int32 arithmetic)
                int current_dist = distances[tour[i] * num_cities + tour[i_next]] + distances[tour[j] * num_cities + tour[j_next]];
                // Distance if we swap the edges
                int new_dist = distances[tour[i] * num_cities + tour[j]] + distances[tour[i_next] * num_cities + tour[j_next]];

                if (new_dist < current_dist) { // Exact integer comparison, no epsilon needed
                    reverse_segment(tour, i + 1, j);
                    improved = true;
                }
            }
        }
    }
}


// --- Kernels ---

// Kernel to initialize random number generator states
extern "C" __global__ void init_rand_states(curandState_t* states, unsigned long long seed, int population_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < population_size) {
        curand_init(seed + tid, 0, 0, &states[tid]);
    }
}

// Kernel to initialize the population with random tours
extern "C" __global__ void initialize_population(
    int* population,
    long long* fitness,
    int* distances,
    int num_cities,
    int population_size,
    curandState_t* rand_states
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < population_size) {
        curandState_t* my_rand_state = &rand_states[tid];
        int* my_tour = &population[tid * num_cities];

        // Create a sequential tour [0, 1, 2, ..., N-1]
        for (int i = 0; i < num_cities; ++i) {
            my_tour[i] = i;
        }

        // Shuffle the tour using Fisher-Yates algorithm
        for (int i = num_cities - 1; i > 0; --i) {
            int j = curand(my_rand_state) % (i + 1);
            int temp = my_tour[i];
            my_tour[i] = my_tour[j];
            my_tour[j] = temp;
        }

        // Calculate initial fitness
        fitness[tid] = calculate_fitness(my_tour, num_cities, distances);
    }
}


extern "C" __global__ void ga_evolution_early_stop(
    int* population,          // Global memory for the current population
    long long* fitness,           // Global memory for fitness values
    int* new_population,      // Global memory buffer for the next generation
    int* distances,
    int num_cities,
    int population_size,
    int max_generations,
    float mutation_rate,
    int elite_size,
    int tournament_size,
    curandState_t* rand_states, // Pre-initialized random states
    int* best_tour,
    long long* best_fitness,
    long long optimal_cost,       // NEW: Optimal cost for early stopping (-1 to disable)
    int patience,             // NEW: Stagnation patience (adaptive: 2.sqrt(n), passed from host)
    int two_opt_iterations,   // 2-opt iterations per individual (ISO-algorithmic)
    int* stopped_generation   // NEW: Output - actual generation stopped at
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= population_size) return;

    curandState_t* my_rand_state = &rand_states[tid];

    // Shared memory for convergence tracking
    __shared__ int converged;
    __shared__ long long prev_best_fitness;
    __shared__ int last_improvement_gen;
    __shared__ int actual_gen;
    
    // Initialize convergence tracking (single thread)
    if (tid == 0) {
        converged = 0;
        prev_best_fitness = 0x7FFFFFFFFFFFFFFFLL;
        last_improvement_gen = 0;
        actual_gen = 0;
    }
    __syncthreads();

    // --- Evolution Loop ---
    for (int gen = 0; gen < max_generations; ++gen) {
        __syncthreads();

        // --- 1. Elitism ---
        if (tid < elite_size) {
            // This assumes a parallel sort has been performed.
            // For now, we just copy the first `elite_size` individuals.
            int* current_tour = &population[tid * num_cities];
            int* next_tour = &new_population[tid * num_cities];
            for (int i = 0; i < num_cities; ++i) {
                next_tour[i] = current_tour[i];
            }
        }
        
        __syncthreads();

        // --- 2. Selection, Crossover, and Mutation ---
        if (tid >= elite_size) {
            // --- Tournament Selection ---
            int p1_idx = -1, p2_idx = -1;
            long long p1_fitness = 0x7FFFFFFFFFFFFFFFLL, p2_fitness = 0x7FFFFFFFFFFFFFFFLL;

            for (int i = 0; i < tournament_size; ++i) {
                int r_idx = curand(my_rand_state) % population_size;
                if (fitness[r_idx] < p1_fitness) {
                    p1_fitness = fitness[r_idx];
                    p1_idx = r_idx;
                }
            }

            for (int i = 0; i < tournament_size; ++i) {
                int r_idx = curand(my_rand_state) % population_size;
                if (r_idx != p1_idx && fitness[r_idx] < p2_fitness) {
                    p2_fitness = fitness[r_idx];
                    p2_idx = r_idx;
                }
            }
            if (p2_idx == -1) p2_idx = (p1_idx + 1) % population_size;

            int* parent1 = &population[p1_idx * num_cities];
            int* parent2 = &population[p2_idx * num_cities];
            int* offspring = &new_population[tid * num_cities];

            // --- Crossover (Order Crossover - OX1) ---
            bool in_offspring[MAX_CITIES]; // Use stack array, requires MAX_CITIES
            for(int i=0; i<num_cities; ++i) in_offspring[i] = false;

            int start = curand(my_rand_state) % num_cities;
            int end = curand(my_rand_state) % num_cities;
            if (start > end) { int temp = start; start = end; end = temp; }

            for (int i = start; i <= end; ++i) {
                offspring[i] = parent1[i];
                in_offspring[parent1[i]] = true;
            }

            int offspring_idx = (end + 1) % num_cities;
            for (int i = 0; i < num_cities; ++i) {
                int p2_gene_idx = (end + 1 + i) % num_cities;
                int gene = parent2[p2_gene_idx];
                if (!in_offspring[gene]) {
                    offspring[offspring_idx] = gene;
                    offspring_idx = (offspring_idx + 1) % num_cities;
                }
            }

            // --- Mutation (Swap Mutation) ---
            if ((curand(my_rand_state) & 255) < (mutation_rate * 255)) {
                int idx1 = curand(my_rand_state) % num_cities;
                int idx2 = curand(my_rand_state) % num_cities;
                int temp = offspring[idx1];
                offspring[idx1] = offspring[idx2];
                offspring[idx2] = temp;
            }

            // --- Local Search (2-Opt) ---
            // Apply 2-Opt to the newly created offspring
            two_opt_device(offspring, num_cities, distances, two_opt_iterations); // Parameterized (was hardcoded 50)
        }

        __syncthreads();

        // --- 3. Update Population and Fitness ---
        if (tid < population_size) {
            int* current_tour = &population[tid * num_cities];
            int* next_tour = &new_population[tid * num_cities];
            for (int i = 0; i < num_cities; ++i) {
                current_tour[i] = next_tour[i];
            }
            fitness[tid] = calculate_fitness(current_tour, num_cities, distances);
        }
        __syncthreads();
        
        // --- Convergence Check (single thread) ---
        if (tid == 0) {
            actual_gen = gen + 1;
            
            // Find current best fitness
            long long curr_best = fitness[0];
            for (int i = 1; i < population_size; ++i) {
                if (fitness[i] < curr_best) curr_best = fitness[i];
            }
            
            // // Check if optimal cost reached
            // if (optimal_cost > 0 && fabsf(curr_best - optimal_cost) < 1e-6f) {
            //     converged = 1;
            // }
            // Check if optimal cost reached (within 1 gap)
            if (optimal_cost > 0) {
                long long gap_times_100 = (curr_best - optimal_cost) * 100 / optimal_cost;
                if (gap_times_100 < 1) {  // Within 1% of optimal
                    converged = 1;
                }
            }
            // Check for improvement (stagnation detection)
            if (curr_best < prev_best_fitness) {
                // Improvement detected
                last_improvement_gen = gen;
                prev_best_fitness = curr_best;
            } else if (gen - last_improvement_gen >= patience) {
                // Stagnation detected
                converged = 1;
            }
        }
        __syncthreads();
        
        // Early exit if converged
        if (converged) break;
    }

    // --- Final Reduction to find the best tour ---
    if (tid == 0) {
        long long overall_best_fitness = fitness[0];
        int best_idx = 0;
        for (int i = 1; i < population_size; ++i) {
            if (fitness[i] < overall_best_fitness) {
                overall_best_fitness = fitness[i];
                best_idx = i;
            }
        }
        *best_fitness = overall_best_fitness;
        for (int i = 0; i < num_cities; ++i) {
            best_tour[i] = population[best_idx * num_cities + i];
        }
        // Write actual generation count (for early stopping tracking)
        *stopped_generation = actual_gen;
    }
}

