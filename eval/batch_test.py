"""
This script compares results of different algorithms. It compares the algorithms' performance on each program that they are evaluated on.
"""
import json
import sys

from test_solution import process_results

result_files = sys.argv[1:]

consider_range = map(str, list(range(0, 5000)))

if len(result_files) == 0:
    print("python batch_test.py [*all_results.json]")
    exit()

eval_results = []
for result_file in result_files:
    try:
        with open(result_file, 'r') as f:
            result = json.load(f)
    except:
        raise Exception(f'fail to load file {result_file}')

    eval_results.append(result)

shared_indices = set(consider_range)
for eval_result in eval_results:
    shared_indices = set(eval_result['times'].keys()) & set(shared_indices)

shared_indices = sorted(list(shared_indices))

for key in ['rewards', 'times']:
    print(key)
    for index in shared_indices:
        print(index, '\t', end='')
        for eval_result in eval_results:
            print("%.4f" % eval_result[key][index], '\t', end='')

        print()

for filename, eval_result in zip(result_files, eval_results):
    print(filename)
    if 'compilation_times' not in eval_result: eval_result['compilation_times'] = {}
    if 'runtime_errors' not in eval_result: eval_result['runtime_errors'] = {}

    process_results(eval_result, filter_indices=shared_indices)
    print()