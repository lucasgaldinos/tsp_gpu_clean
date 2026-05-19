import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def objective_function(x):
    return math.sin(x) + math.cos(2 * x)

def get_neighbors(x, lower_bound, upper_bound):
    neighbors = []
    if x - 1 >= lower_bound:
        neighbors.append(x - 1)
    if x + 1 <= upper_bound:
        neighbors.append(x + 1)
    return neighbors

def tabu_search_animated(
    objective_function,
    initial_solution,
    tabu_size,
    max_iterations,
    lower_bound,
    upper_bound,
    ax1,
    ax2
):
    current_solution = initial_solution
    best_solution = current_solution
    best_value = objective_function(current_solution)

    tabu_list = [current_solution]

    history = [best_value]
    path = [current_solution]

    x_values = list(range(lower_bound, upper_bound + 1))
    y_values = [objective_function(x) for x in x_values]

    # Initialize plots
    line1, = ax1.plot(x_values, y_values, label='f(x)')
    visited_scatter, = ax1.plot([], [], 'ro', label='Visited')
    best_scatter, = ax1.plot([], [], 'go', label='Best')
    ax1.legend()
    ax1.set_title('Tabu Search Progress')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True)

    line2, = ax2.plot([], [], 'b-')
    ax2.set_title('Convergence History')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best f(x) Found')
    ax2.grid(True)

    def update(frame):
        nonlocal current_solution, best_solution, best_value

        if frame >= max_iterations:
            return line1, visited_scatter, best_scatter, line2

        neighbors = get_neighbors(current_solution, lower_bound, upper_bound)
        
        best_neighbor = None
        best_neighbor_value = -math.inf

        for neighbor in neighbors:
            neighbor_value = objective_function(neighbor)
            if (neighbor not in tabu_list) or (neighbor_value > best_value):
                if neighbor_value > best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

        if best_neighbor is None:
            print("No non-tabu neighbors found. Terminating search.")
            ani.event_source.stop()
            return line1, visited_scatter, best_scatter, line2

        current_solution = best_neighbor
        path.append(current_solution)

        if best_neighbor_value > best_value:
            best_solution = best_neighbor
            best_value = best_neighbor_value
            print(f"Iteration {frame+1}: New best solution x = {best_solution}, f(x) = {best_value:.4f}")

        history.append(best_value)

        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        # Update plots
        visited_scatter.set_data(path, [objective_function(x) for x in path])
        best_scatter.set_data(best_solution, best_value)
        line2.set_data(range(len(history)), history)
        ax2.set_xlim(0, max_iterations)
        ax2.set_ylim(min(history) - 0.1, max(history) + 0.1)

        return line1, visited_scatter, best_scatter, line2

    ani = animation.FuncAnimation(
        plt.gcf(),
        update,
        frames=max_iterations,
        interval=200,
        repeat=False
    )

    plt.show()

    return best_solution, best_value, history, path

def main_animated():
    # Parameters
    lower_bound = 0
    upper_bound = 20
    initial_solution = random.randint(lower_bound, upper_bound)
    tabu_size = 5
    max_iterations = 50

    print(f"Initial solution: x = {initial_solution}, f(x) = {objective_function(initial_solution):.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    best_x, best_fx, history, path = tabu_search_animated(
        objective_function,
        initial_solution,
        tabu_size,
        max_iterations,
        lower_bound,
        upper_bound,
        ax1,
        ax2
    )

    print(f"\nBest solution found: x = {best_x}, f(x) = {best_fx:.4f}")

if __name__ == "__main__":
    main_animated()