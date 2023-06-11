"""
Print the evaluation results for finished runs.
"""
import json

import numpy as np
import os
import pprint

from tqdm import tqdm


def process_results(results, save=None, filter_indices=None):
    """
    Print the results and log, also return it possibly for analysis
    """
    rewards, rewards_train, times, sample_times, compile_errors, runtime_errors = \
        results['rewards'], results['rewards_train'], results['times'], results['sample_times'], results['compile_errors'], results['runtime_errors']
    if filter_indices is not None:
        filter_op = lambda result: {k: result[k] for k in result.keys() if k in filter_indices}
    else:
        filter_op = lambda result: result
    # ignore the keys and get the values, and convert to np.array
    convert_to_array = lambda result: np.array(list(result.values()))

    # filter all dictionaries by filter_indices, and convert them to numpy array
    rewards, rewards_train, times, sample_times, compile_errors, runtime_errors =\
        map(lambda x: convert_to_array(filter_op(x)), [rewards, rewards_train, times, sample_times, compile_errors, runtime_errors])

    print(f"Got {len(times)} programs, averaging over {len(rewards)} programs")

    stat = {
        'result_dir': save,
        'pass_rate': 100 * np.mean(rewards),
        'ac': 100 * np.mean(np.array(rewards) == 1.),
        'training_pass_rate': 100 * np.mean(rewards_train) if None not in rewards_train else None,
        'training_ac': 100 * np.mean(rewards_train == 1.) if None not in rewards_train else None,
        'time': np.mean(times),
        'sample_times': np.mean(sample_times),
        'avg_compile_errors': 100 * np.mean(compile_errors),
        'has_compile_errors': 100 * np.mean(compile_errors > 0),
        'avg_runtime_errors': 100 * np.mean(runtime_errors),
        'has_runtime_errors': 100 * np.mean(runtime_errors > 0),
    }
    print("On the test set")
    print(f"Test Case Average (average accuracy over problems) = {stat['pass_rate']} %")
    print(f"Strict Accuracy (all test cases passed / total problems) = {stat['ac']} %")
    print()

    print("On the training set")
    print(f"Test Case Average (average accuracy over problems) = {stat['training_pass_rate']} %")
    print(f"Strict Accuracy (all test cases passed / total problems) = {stat['training_ac']} %")
    print()

    print('avg compile errors', stat['avg_compile_errors'])
    print('has compile errors', stat['has_compile_errors'])
    print('avg runtime errors', stat['avg_runtime_errors'])
    print('has runtime errors', stat['has_runtime_errors'])

    if len(times) > 0:
        print(f"Computation time (sec.): Mean: {stat['time']}, SE: {np.std(times) / np.sqrt(len(times))}, "
              f"Max: {np.max(times)}, Min: {np.min(times)}")
    else:
        print("No computation time recorded, or none was successfully run.")

    if len(sample_times) > 0:
        print(f"Sample times: Mean: {stat['sample_times']}, SE: {np.std(sample_times) / np.sqrt(len(sample_times))}, "
              f"Max: {np.max(sample_times)}, Min: {np.min(sample_times)}")
    else:
        print("No sample times recorded.")

    return stat


def eval_and_save_problems(args, indices, k, result_prefix):
    """
    Args:
        args: command arguments
        indices: the indices of problems to be evaluated
        result_prefix: add this to the output file (may denote top-k, top-n, etc.)
    """
    all_results_loc = os.path.join(args.save, f"{result_prefix}all_results.json")
    try:
        with open(all_results_loc, 'r') as f:
            all_results = json.load(f)

        rewards, rewards_train, times, sample_times, compile_errors, runtime_errors = \
            all_results['rewards'], all_results['rewards_train'], all_results['times'], all_results['sample_times'], all_results['compile_errors'], all_results['runtime_errors']
    except:
        print(f"{all_results_loc} specified, but failed to open.")
        rewards, rewards_train, times, sample_times, compile_errors, runtime_errors = {}, {}, {}, {}, {}, {}

    # don't show progress bar if only one problem to test
    indices = tqdm(indices) if len(indices) > 1 else indices

    for index in indices:
        if str(index) in rewards.keys() and not args.retest:
            print(f"skipping {index} because it's already in results json file")
            continue

        code_loc = os.path.join(args.save, f"{args.prefix}{index}.json")

        if not os.path.exists(code_loc):
            print(f"didn't find result for {index}")
            continue
        else:
            # result_loc exists, simply read it
            try:
                with open(code_loc) as f:
                    result = json.load(f)

                    reward, reward_train, time, sample_time = result['rewards'], result['train rewards'],\
                                                              result['time'], result['sample times']
            except Exception as e:
                print(f"failed to read {code_loc}, {e}")
                continue

        if len(reward) == 0 or (isinstance(time, list) and len(time) == 0):
            print(f"{code_loc} results non-parsable.")
            continue

        # in the setting where we only use k generated samples, use the first k of all samples
        if k is not None:
            # if k is smaller than generated samples, find computation time proportional to k / sample times
            if isinstance(time, list):
                if k < len(time):
                    time = time[k]
                else:
                    time = time[-1]
            else:
                # total time
                time = min(k / sample_time, 1) * time

            reward = reward[:k]
            reward_train = reward_train[:k]
        else:
            if isinstance(time, list):
                time = time[-1]

        if len(reward_train) > 0:
            # sort the training rewards of the samples, get the top n of them
            top_n_indices = np.argsort(reward_train)[::-1][:args.n]
            # find the one that has the highest test reward
            return_index = max(top_n_indices, key=lambda x: reward[x])
        else:
            return_index = 0

        # add to the list
        rewards[index] = reward[return_index]
        rewards_train[index] = reward_train[return_index] if len(reward_train) > 0 else 0
        # these values are None for failed experiments
        if time is not None: times[index] = time

        if k is not None:
            # use first k number if k is smaller than sample times
            sample_time = min(k, sample_time)
        sample_times[index] = sample_time

        try:
            compile_errors[index] = result['compile_error']
            runtime_errors[index] = result['runtime_error']
        except:
            compile_errors[index] = 0
            runtime_errors[index] = 0

    # save results to file
    all_results = {
        'rewards': rewards,
        'rewards_train': rewards_train,
        'times': times,
        'sample_times': sample_times,
        'compile_errors': compile_errors,
        'runtime_errors': runtime_errors
    }

    with open(all_results_loc, "w") as f:
        try:
            json.dump(all_results, f)
        except Exception as e:
            print(f"Couldn't save all results.\n{e}")

    # return results from args.start to args.end
    filter_op = lambda x: {k: x[k] for k in x.keys() if int(k) in indices}
    ret_results = {k: filter_op(v) for k, v in all_results.items()}

    return ret_results


def main(arg_dict=None):
    import argparse

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t","--test-loc", default="../data_split/test.json", type=str, help="path to the json containing problem paths to be evaluated.")
    parser.add_argument("-r","--root", default="", type=str, help="where the data is stored.")
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("--indices", default=None, type=str, help="Read a list of indices from a json file.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results/", help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix of generated code file.")

    # the following implements the n@k metric
    # k is the number of samples. By default, k is None, where we use all the samples we have
    # n is the number of submissions. By default, n = 1, where it evaluates the problem with the highest training reward
    # k-list example: 128,256,512,null
    parser.add_argument("--k-list", type=str, default='null', help='Find the best program in the first k generated programs.')
    parser.add_argument("-n", default=1, type=int, help='Evaluate using the n best program candidates (the n programs that have the highest pass rate on the training set.')

    parser.add_argument('--retest', action='store_true', default=False, help="rerun tests.")

    parser.add_argument("--public-cases", type=str, default='half')

    parser.add_argument('--wandb', action='store_true', default=False, help="log results to a wandb run")
    parser.add_argument('--wandb-id', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)

    if arg_dict is not None:
        arg_list = []
        for k, v in arg_dict.items():
            arg_list.append(f"--{k}")
            arg_list.append(str(v))
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # get indices to evaluate
    if args.end is not None:
        # if end is passed, only evaluate problems within the range
        indices = range(args.start, args.end)
    elif args.index is not None:
        indices = [args.index]
    elif args.indices is not None:
        with open(args.indices) as f:
            indices = json.load(f)
    else:
        raise Exception("Invalid index")

    n_flag = f"n{args.n}-" if args.n > 1 else ""

    stats = dict()
    for k in args.k_list.split(','):
        if k.isdigit():
            k = int(k)
        else:
            k = None

        k_flag = f"k{k}-" if k is not None else ""
        result_prefix = f"{args.prefix}{k_flag}{n_flag}"

        results = eval_and_save_problems(args, indices, k, result_prefix)

        stat = process_results(results, args.save)
        stats[k] = stat

    return stats


if __name__ == "__main__":
    main()