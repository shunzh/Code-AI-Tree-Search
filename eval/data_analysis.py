"""
This script is used to analyze the data and print the number of public / private test cases for the datasets.
"""
import collections
import json
import os
from matplotlib import pyplot as plt
import numpy as np

from tqdm import tqdm

#test_loc = "../CodeContest_data_split/test.json"
#test_loc = "../data_split/test.json"
test_loc = "../data_split/train.json"
public_io_lengths = []
io_lengths = []
solution_lengths = []

with open(test_loc, "r") as f:
    problems = json.load(f)

for problem in tqdm(problems):
    prob_path = os.path.join('', problem)
    public_test_case_path = os.path.join(prob_path, "public_input_output.json")
    test_case_path = os.path.join(prob_path, "input_output.json")
    solution_path = os.path.join(prob_path, "solutions.json")

    try:
        with open(public_test_case_path, "r") as f:
            public_test_cases = json.load(f)
            public_io_lengths.append(len(public_test_cases["inputs"]))
    except:
        print('public input_output not found')

    try:
        with open(test_case_path, "r") as f:
            test_cases = json.load(f)
            io_lengths.append(len(test_cases["inputs"]))
    except:
        print('input_output not found')

    try:
        with open(solution_path, "r") as f:
            solutions = json.load(f)
            solution_lengths.append(len(solutions))
    except:
        print('solution not found')
        solution_lengths.append(0)

print('solutions', np.mean(solution_lengths))

print('public ios', np.mean(public_io_lengths))
print('ios', np.mean(io_lengths))
print(collections.Counter(io_lengths))
plt.scatter(range(len(io_lengths)), io_lengths)
plt.show()
