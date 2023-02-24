import multiprocessing

import testing_util as test_util
import numpy as np

"""
Running set in a separate process
https://github.com/hendrycks/apps/blob/83d925041b1c43c32b56d444bb315f729f4ff633/eval/test_one_solution.py#L57
"""
def _temp_run(prob_path, output_str, mode, public_test_cases, result):
    result.append(test_util.run_test(prob_path=prob_path, test=output_str, mode=mode, public_test_cases=public_test_cases))

def check_correctness(prob_path, output_str, mode, public_test_cases):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(prob_path, output_str, mode, public_test_cases, result))
    p.start()
    p.join(timeout=10)
    if p.is_alive():
        p.kill()
    if not result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
    return result[0]


def compute_reward(prob_path, output_str, mode='train', public_test_cases=None, return_info=False):
    """
    A utility function that computes the reward given problem path and output string of our model
    It is rewarded by the number of tests passed. When passing the same number of tests.
    """
    # from https://github.com/hendrycks/apps/blob/83d925041b1c43c32b56d444bb315f729f4ff633/eval/test_one_solution.py#L141
    try:
        curr_res = check_correctness(prob_path, output_str, mode, public_test_cases)
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
        # if not np.all(curr_res):
        #     print(f"Results were not all True: {curr_res}")
    except Exception as e:
        print(f"test framework exception = {repr(e)}{e}\n")
        curr_res = []

    # How to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
    assert isinstance(curr_res, list)
    pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0

    if return_info:
        info = {"compile_error": curr_res.count(-2) / len(curr_res), "runtime_error": curr_res.count(-1) / len(curr_res)}
        return pass_rate, info
    else:
        return pass_rate

def get_program_quality(s):
    """
    For now, only consider the length of the program. The shorter, the better.
    """
    return np.exp(- len(s) / 20)

