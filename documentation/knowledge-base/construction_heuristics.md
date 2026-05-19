---
date: "2025-11-25"
title: "11 Animated Algorithms for the Traveling Salesman Problem"
description: "STEM Lounge offers in-depth explanations and animated visualizations of various topics across science, technology, engineering, and mathematics. It explores complex algorithms, data visualizations, and scientific concepts, making them accessible and understandable through engaging content."
---
## TSP Algorithms and heuristics

Although we haven’t been able to quickly find optimal solutions to NP problems like the Traveling Salesman Problem, "good-enough" solutions to NP problems can be quickly found <sup>[1]</sup>.

For the visual learners, here’s an animated collection of some well-known heuristics and algorithms in action. Researchers often use these methods as sub-routines for their own algorithms and heuristics. This is not an exhaustive list.

For ease of visual comparison we use `Dantzig49` as the common TSP problem, in Euclidean space. Dantzig49 has 49 cities — one city in each contiguous US State, plus Washington DC.

![Dantzig49](https://stemlounge.com/content/images/size/w600/2021/06/dantzig49.png)

![Dantzig49 optimum](https://stemlounge.com/content/images/2021/06/d49_optimum.gif)

The optimum solution to `Dantzig49`. One of $6,206,957,796,268,036,335,431,144,523,686,687,519,260,743,177,338,880,000,000,000 = (49-1)!$ possible tours

## 1: Greedy Algorithm

A greedy algorithm is a general term for algorithms that try to add the lowest cost possible in each iteration, even if they result in sub-optimal combinations.

In this example, all possible edges are sorted by distance, shortest to longest. Then the shortest edge that will neither create a vertex with more than 2 edges, nor a cycle with less than the total number of cities is added. This is repeated until we have a cycle containing all of the cities.

![Greedy Algorithm](https://stemlounge.com/content/images/2021/06/greedy_algorithm.gif)

Greedy Algorithm

Although all the heuristics here cannot guarantee an optimal solution, greedy algorithms are known to be especially sub-optimal for the TSP.

## 2: Nearest Neighbor

The nearest neighbor heuristic is another greedy algorithm, or what some may call naive. It starts at one city and connects with the closest unvisited city. It repeats until every city has been visited. It then returns to the starting city.

![Nearest Neighbor](https://stemlounge.com/content/images/2021/06/nearest_neighbor.gif)

Nearest Neighbor Heuristic

Karl Menger, who first defined the TSP, noted that nearest neighbor is a sub-optimal method:

> "The rule that one first should go from the staring point to the closest point, then to the point closest to this, etc., in general does not yield the shortest route."

The time complexity of the nearest neighbor algorithm is `O(n^2)`. The number of computations required will not grow faster than n^2.

## 3: Nearest Insertion

Insertion algorithms add new points between existing points on a tour as it grows.

One implementation of Nearest Insertion begins with two cities. It then repeatedly finds the city not already in the tour that is closest to any city in the tour, and places it between whichever two cities would cause the resulting tour to be the shortest possible. It stops when no more insertions remain.

![Nearest Insertion](https://stemlounge.com/content/images/2021/06/nearest_insertion.gif)

Nearest Insertion

The nearest insertion algorithm is `O(n^2)`

## 4: Cheapest Insertion

Like Nearest Insertion, Cheapest Insertion also begins with two cities. It then finds the city not already in the tour that when placed between two connected cities in the subtour will result in the shortest possible tour. It inserts the city between the two connected cities, and repeats until there are no more insertions left.

![Cheapest insertion](https://stemlounge.com/content/images/2021/06/cheapest_insertion.gif)

Cheapest Insertion

The cheapest insertion algorithm is `O(n^2 log2(n))`

## 5: Random Insertion

Random Insertion also begins with two cities. It then randomly selects a city not already in the tour and inserts it between two cities in the tour. Rinse, wash, repeat.

![Random Insertion](https://stemlounge.com/content/images/2021/06/random_insertion.gif)

Random Insertion

Time complexity: `O(n^2)`

## 6: Farthest Insertion

Unlike the other insertions, Farthest Insertion begins with a city and connects it with the city that is furthest from it.

It then repeatedly finds the city not already in the tour that is furthest from any city in the tour, and places it between whichever two cities would cause the resulting tour to be the shortest possible.

![Farthest insertion](https://stemlounge.com/content/images/2021/06/farthest_insertion.gif)

Farthest Insertion

Time complexity: `O(n^2)`

## 7: Christofides Algorithm

Christofides algorithm is a heuristic with a 3/2 approximation guarantee. In the worst case the tour is no longer than 3/2 the length of the optimum tour.

Due to its speed and 3/2 approximation guarantee, Christofides algorithm is often used to construct an upper bound, as an initial tour which will be further optimized using tour improvement heuristics, or as an upper bound to help limit the search space for branch and cut techniques used in search of the optimal route.

For it to work, it requires distances between cities to be symmetric and obey the triangle inequality, which is what you'll find in a typical x,y coordinate plane (metric space). Published in 1976, it continues to hold the record for the best approximation ratio for metric space.

![Christofides Algorithm](https://stemlounge.com/content/images/2021/06/christofides.gif)

Christofides Algorithm

The algorithm is intricate. Its time complexity is `O(n^4)`

## 8: 2-Opt

A problem is called k-Optimal if we cannot improve the tour by switching k edges.

Each k-Opt iteration takes `O(n^k)` time.

2-Opt is a local search tour improvement algorithm proposed by Croes in 1958. It originates from the idea that tours with edges that cross over aren’t optimal. 2-opt will consider every possible 2-edge swap, swapping 2 edges when it results in an improved tour.

![2-opt](https://stemlounge.com/content/images/2021/06/2-opt.gif)

2-Opt

2-opt takes `O(n^2)` time per iteration.

## 9: 3-Opt

3-opt is a generalization of 2-opt, where 3 edges are swapped at a time. When 3 edges are removed, there are 7 different ways of reconnecting them, so they're all considered.

![3-opt](https://stemlounge.com/content/images/2021/06/3-opt.gif)

3-Opt

The time complexity of 3-opt is `O(n^3)` for every 3-opt iteration.

## 10: Lin-Kernighan Heuristic

Lin-Kernighan is an optimized k-Opt tour-improvement heuristic. It takes a tour and tries to improve it.

By allowing some of the intermediate tours to be more costly than the initial tour, Lin-Kernighan can go well beyond the point where a simple 2-Opt would terminate <sup>[4]</sup>.

![LKH](https://stemlounge.com/content/images/2021/06/lin_kernighan.gif)

LKH

Implementations of the Lin-Kernighan heuristic such as Keld Helsgaun's LKH may use "walk" sequences of 2-Opt, 3-Opt, 4-Opt, 5-Opt, “kicks” to escape local minima, sensitivity analysis to direct and restrict the search, as well as other methods.

LKH has 2 versions; the original and LKH-2 released later. Although it's a heuristic and not an exact algorithm, it frequently produces optimal solutions. It has converged upon the optimum route of every tour with a known optimum length. At one point in time or another it has also set records for every problem with unknown optimums, such as the World TSP, which has 1,900,000 locations.
