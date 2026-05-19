import json, glob, numpy as np

for prob in ['berlin52', 'kroA100', 'pr1002']:
    with open(f'tmp/stats_output/problem_statistics/{prob}_stats.json') as f:
        data = json.load(f)
        print(f"Problem: {prob}")
        # Wait, problem_statistics only has pairwise comparisons and friedman, not raw times!
