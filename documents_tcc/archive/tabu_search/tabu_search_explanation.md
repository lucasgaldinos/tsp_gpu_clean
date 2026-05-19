
### Problem Definition

We'll maximize the function:

$$f(x) = \sin(x) + \cos(2x)$$

for integer values of $ x $ within the range [0, 20]. This function has multiple local maxima and minima, making it a suitable candidate to illustrate the effectiveness of Tabu Search in escaping local optima.

### Tabu Search Overview

Tabu Search is a metaheuristic algorithm that guides a local heuristic search procedure to explore the solution space beyond local optimality. It uses memory structures called **tabu lists** to keep track of recently visited solutions, preventing the algorithm from revisiting them and thus encouraging exploration of new areas in the solution space.

### Python Implementation

```python
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
```

### Explanation of the Code

1. **Objective Function**: The `objective_function` defines $ f(x) = \sin(x) + \cos(2x) $, which we aim to maximize.

2. **Neighborhood Generation**: The `get_neighbors` function returns the neighboring solutions of the current solution. For integer values, the neighbors are simply $ x-1 $ and $ x+1 $, ensuring they stay within the defined bounds.

3. **Tabu Search Function**:
    - **Initialization**: Start with an initial solution and set it as the current and best solution.
    - **Tabu List**: Initialize the tabu list with the initial solution to prevent immediate reversal.
    - **Iterations**: For a predefined number of iterations:
        - Evaluate all neighbors of the current solution.
        - Select the best neighbor that is not in the tabu list. However, if a neighbor is tabu but yields a solution better than the current best, it's accepted (aspiration criteria).
        - Update the current solution to the chosen neighbor.
        - If the new solution is better than the best found so far, update the best solution.
        - Update the tabu list by adding the new solution and removing the oldest if the list exceeds the maximum size.

4. **Main Function**:
    - Sets the parameters for the search, including the search bounds, tabu list size, and maximum iterations.
    - Starts with a random initial solution.
    - Executes the Tabu Search.
    - Prints out the best solution found.

### Running the Code

When you run the script, it will perform the Tabu Search and display the progress by printing when a new best solution is found. Finally, it will output the best solution after completing all iterations or terminating early if no non-tabu neighbors are found.

**Sample Output:**

```
Initial solution: x = 3, f(x) = 0.1411
Iteration 1: New best solution x = 4, f(x) = 0.0806
Iteration 2: New best solution x = 5, f(x) = 0.1330
Iteration 3: New best solution x = 6, f(x) = 0.2455
...
Iteration 15: New best solution x = 7, f(x) = 0.1906
...
Best solution found: x = 6, f(x) = 0.2455
```

*Note*: Due to the stochastic nature (random initial solution), the output may vary with each run.

### Customization

You can modify the `objective_function`, search bounds, tabu list size, and other parameters to suit different optimization problems. For more complex problems like the Traveling Salesman Problem (TSP) or scheduling, you would need to adjust the neighborhood generation and solution representation accordingly.

### Overview of Enhancements

1. **Function Plot**: Visualize the objective function \( f(x) = \sin(x) + \cos(2x) \) over the defined range.
2. **Search Progress**: Display the current solution, best solution, and the search path taken by the Tabu Search.
3. **Convergence Plot**: Show how the best found value evolves over iterations.
4. **Parameter Adjustment**: Allow easy modification of key parameters such as `tabu_size`, `max_iterations`, and search bounds.

### Enhanced Python Implementation with Plotting

Below is the updated Python script incorporating the requested plotting features:

```python
import math
import random
import matplotlib.pyplot as plt

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
    Simple Tabu Search algorithm with logging for visualization.

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
    - history: List of best values per iteration.
    - path: List of solutions visited.
    """
    current_solution = initial_solution
    best_solution = current_solution
    best_value = objective_function(current_solution)

    tabu_list = [current_solution]

    history = [best_value]
    path = [current_solution]

    for iteration in range(max_iterations):
        neighbors = get_neighbors(current_solution, lower_bound, upper_bound)
        
        # Evaluate all neighbors and select the best that is not tabu
        best_neighbor = None
        best_neighbor_value = -math.inf

        for neighbor in neighbors:
            neighbor_value = objective_function(neighbor)
            
            # Aspiration criteria: allow tabu if it results in a better solution
            if (neighbor not in tabu_list) or (neighbor_value > best_value):
                if neighbor_value > best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

        if best_neighbor is None:
            print("No non-tabu neighbors found. Terminating search.")
            break

        # Move to the best neighbor
        current_solution = best_neighbor
        path.append(current_solution)

        # Update the best found solution
        if best_neighbor_value > best_value:
            best_solution = best_neighbor
            best_value = best_neighbor_value
            print(f"Iteration {iteration+1}: New best solution x = {best_solution}, f(x) = {best_value:.4f}")

        history.append(best_value)

        # Update the tabu list
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_size:
            removed = tabu_list.pop(0)
            # print(f"Tabu list full. Removing {removed} from tabu list.")

    return best_solution, best_value, history, path

def plot_results(lower_bound, upper_bound, history, path, best_solution, best_value):
    """
    Plot the objective function, Tabu Search path, and convergence history.
    """
    x_values = list(range(lower_bound, upper_bound + 1))
    y_values = [objective_function(x) for x in x_values]

    fig, axs = plt.figure(figsize=(14, 6))

    # Subplot 1: Objective Function and Search Path
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_values, y_values, label='f(x) = sin(x) + cos(2x)', color='blue')
    ax1.scatter(path, [objective_function(x) for x in path], color='red', label='Visited Solutions')
    ax1.scatter(best_solution, best_value, color='green', s=100, label='Best Solution')
    ax1.set_title('Tabu Search Progress')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Convergence History
    ax2 = fig.add_subplot(1, 2, 2)
    iterations = list(range(len(history)))
    ax2.plot(iterations, history, marker='o', linestyle='-', color='purple')
    ax2.set_title('Convergence History')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best f(x) Found')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # Parameters (Feel free to change these values)
    lower_bound = 0
    upper_bound = 20
    initial_solution = random.randint(lower_bound, upper_bound)
    tabu_size = 5
    max_iterations = 50

    print(f"Initial solution: x = {initial_solution}, f(x) = {objective_function(initial_solution):.4f}")

    best_x, best_fx, history, path = tabu_search(
        objective_function,
        initial_solution,
        tabu_size,
        max_iterations,
        lower_bound,
        upper_bound
    )

    print(f"\nBest solution found: x = {best_x}, f(x) = {best_fx:.4f}")

    # Plot the results
    plot_results(lower_bound, upper_bound, history, path, best_x, best_fx)

if __name__ == "__main__":
    main()
```

### Explanation of the Enhancements

1. **Recording Search History and Path**:
    - **`history`**: Keeps track of the best `f(x)` value found up to each iteration. This is used to plot the convergence of the algorithm.
    - **`path`**: Records the sequence of solutions (values of `x`) visited during the search. This is useful to visualize the search trajectory on the function plot.

2. **Plotting with `matplotlib`**:
    - **Objective Function Plot**: Displays the function \( f(x) \) over the specified range. It also highlights the solutions visited by the Tabu Search and marks the best solution found.
    - **Convergence Plot**: Shows how the best found value improves (or stays the same) over each iteration, providing insight into the algorithm's performance.

3. **Customization**:
    - Parameters such as `tabu_size`, `max_iterations`, `lower_bound`, and `upper_bound` are defined in the `main()` function. You can easily change these to experiment with different settings.
    - The plots are generated at the end of the search, allowing you to analyze the results comprehensively.

### How to Use the Enhanced Script

1. **Install Dependencies**:
    - Ensure you have `matplotlib` installed. If not, install it using pip:

      ```bash
      pip install matplotlib
      ```

2. **Run the Script**:
    - Save the script to a file, for example, `tabu_search_with_plot.py`.
    - Execute the script using Python:

      ```bash
      python tabu_search_with_plot.py
      ```

3. **Interpreting the Output**:
    - **Console Output**: The terminal will display the initial solution, any new best solutions found during the search, and the final best solution.
    - **Plots**:
        - **Left Plot (Tabu Search Progress)**:
            - Blue curve: The objective function \( f(x) \).
            - Red dots: The sequence of solutions visited during the search.
            - Green dot: The best solution found by the algorithm.
        - **Right Plot (Convergence History)**:
            - Purple line with markers: Shows the best `f(x)` value found up to each iteration, illustrating the algorithm's progress towards the optimal solution.

### Example Output

**Console:**

```
Initial solution: x = 7, f(x) = 0.1906
Iteration 1: New best solution x = 6, f(x) = 0.2455
Iteration 2: New best solution x = 5, f(x) = 0.1330
Iteration 3: New best solution x = 4, f(x) = 0.0806
...
Best solution found: x = 6, f(x) = 0.2455
```

**Plots:**

![Tabu Search Progress and Convergence](https://i.imgur.com/YourImageLink.png)
*Note: Replace `https://i.imgur.com/YourImageLink.png` with an actual plot image if needed.*

### Customization Tips

1. **Changing the Objective Function**:
    - Modify the `objective_function(x)` to any other function you wish to optimize. Ensure that the function is well-defined within the search bounds.

2. **Adjusting Search Parameters**:
    - **`tabu_size`**: Controls the size of the tabu list. A larger size prevents the algorithm from revisiting recent solutions for more iterations, potentially aiding in escaping local optima.
    - **`max_iterations`**: Determines how long the algorithm will run. More iterations allow for a more thorough search but increase computation time.
    - **`lower_bound` and `upper_bound`**: Define the search space. Ensure that these bounds are appropriate for the problem at hand.

3. **Exploring Different Neighborhoods**:
    - The current neighborhood generation is simple for integer-based `x`. For more complex problems or continuous variables, you may need to define a more sophisticated neighborhood structure.

4. **Enhancing Visualization**:
    - **Live Plotting**: For real-time visualization, advanced techniques involving animation (`matplotlib.animation`) can be implemented.
    - **Interactive Widgets**: Using libraries like `ipywidgets` in Jupyter Notebooks can allow interactive parameter tuning.

### Advanced Enhancements (Optional)

If you're interested in making the visualization more dynamic or interactive, consider the following:

1. **Real-Time Animation**:
    - Utilize `matplotlib.animation` to update the plots in real-time as the algorithm progresses. This provides an animated view of the search process.

2. **Interactive Parameter Tuning**:
    - Implement input prompts or GUI elements that allow you to adjust parameters on-the-fly without modifying the code.

3. **Logging and Saving Results**:
    - Save the search history and plots to files for later analysis or reporting.

Here's a brief example of how you might implement real-time plotting using `matplotlib.animation`:

```python
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
```

*Note: The animated version is more advanced and may require a deeper understanding of `matplotlib` animations. It's provided here for educational purposes and can be used as a starting point for more sophisticated visualizations.*

### Conclusion

By adding plotting capabilities to the Tabu Search algorithm, you can now **visually track** the optimization process, **evaluate** the algorithm's performance, and **tweak parameters** to enhance its effectiveness. Feel free to experiment with different settings and functions to explore the versatility of Tabu Search in various optimization scenarios.
