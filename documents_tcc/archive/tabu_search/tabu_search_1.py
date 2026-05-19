import math
import random

def objective_function(x):
    """
    Objective function to maximize: f(x) = sin(x) + cos(2x)
    """
    return math.sin(x) + math.cos(2 * x)

def get_neighbors(x, lower_bound, upper_bound):
    """
    Generate neighboring solutions. For integer x, neighbors are x-1 and x+1.
    """
    neighbors = []
    if x - 1 >= lower_bound:
        neighbors.append(x - 1)
    if x + 1 <= upper_bound:
        neighbors.append(x + 1)
    return neighbors

def tabu_search(
    objective_function,
    initial_solution,
    tabu_size,
    max_iterations,
    lower_bound,
    upper_bound
):
    """
    Simple Tabu Search algorithm.

    Parameters:
    - objective_function: The function to maximize.
    - initial_solution: Starting point for the search.
    - tabu_size: The maximum size of the tabu list.
    - max_iterations: Number of iterations to perform.
    - lower_bound: Lower bound of the search space.
    - upper_bound: Upper bound of the search space.

    Returns:
    - best_solution: The best solution found.
    - best_value: The value of the best solution.
    """
    current_solution = initial_solution
    best_solution = current_solution
    best_value = objective_function(current_solution)

    tabu_list = [current_solution]

    for iteration in range(max_iterations):
        neighbors = get_neighbors(current_solution, lower_bound, upper_bound)
        
        # Evaluate all neighbors and select the best that is not tabu
        best_neighbor = None
        best_neighbor_value = -math.inf

        for neighbor in neighbors:
            neighbor_value = objective_function(neighbor)
            
            if neighbor_value > best_neighbor_value:
                if neighbor not in tabu_list or neighbor_value > best_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

        if best_neighbor is None:
            print("No non-tabu neighbors found. Terminating search.")
            break

        # Move to the best neighbor
        current_solution = best_neighbor

        # Update the best found solution
        if best_neighbor_value > best_value:
            best_solution = best_neighbor
            best_value = best_neighbor_value
            print(f"Iteration {iteration+1}: New best solution x = {best_solution}, f(x) = {best_value:.4f}")

        # Update the tabu list
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_size:
            removed = tabu_list.pop(0)
            # print(f"Tabu list full. Removing {removed} from tabu list.")

    return best_solution, best_value

def main():
    # Parameters
    lower_bound = 0
    upper_bound = 20
    initial_solution = random.randint(lower_bound, upper_bound)
    tabu_size = 5
    max_iterations = 50

    print(f"Initial solution: x = {initial_solution}, f(x) = {objective_function(initial_solution):.4f}")

    best_x, best_fx = tabu_search(
        objective_function,
        initial_solution,
        tabu_size,
        max_iterations,
        lower_bound,
        upper_bound
    )

    print(f"\nBest solution found: x = {best_x}, f(x) = {best_fx:.4f}")

if __name__ == "__main__":
    main()