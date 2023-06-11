import json
import os
import pprint
import sys
import time

import torch
import transformers
import numpy as np
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../eval/')

from default_pi import APPSHeuristic

from transformer_utils.utils import get_model_by_name

# okay with parallelization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# required by huggingface code_eval
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def bs_exp(args, env, dp):
    """
    Run beam search
    """
    s = env.state
    s = dp.get_predict_sequence(s, horizon=args.horizon)
    return [s], {'sample_times': args.num_beams}

def sample_exp(args, env, dp):
    """
    Run sampling + filtering
    """
    s = env.state

    assert dp.ts_mode == 'sample' # this should be specified after sampling alg is specified
    samples = [dp.get_predict_sequence(s, horizon=args.horizon) for _ in tqdm(range(args.num_samples))]
    return samples, {'sample_times': args.num_samples}


def main():
    if args.index is not None:
        problem_indices = [args.index]
    elif args.end is not None:
        problem_indices = range(args.start, args.end)
    elif args.indices is not None:
        # customized list (maybe cases that have high performances under Transformer)
        with open(args.indices) as f:
            problem_indices = json.load(f)
    else:
        raise Exception("Don't know what problems to solve.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Loading model {args.load}")
    model, tokenizer = get_model_by_name(args.load, args.device)
    print("Model loaded/initialized.")

    if args.load_value is not None:
        print(f"Loading value model {args.load_value}")
        value_model = transformers.GPT2ForSequenceClassification.from_pretrained(args.load_value)
        print("Value model loaded.")
    else:
        value_model = None

    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)

    # pre-processing dataset
    if args.dataset == 'apps':
        # get problem locations
        with open(args.test_loc, "r") as f:
            problems = json.load(f)
        # get a list of program file paths
        problems = [problems[idx] for idx in problem_indices]
    else:
        raise Exception(f"Unknown dataset {args.dataset}")

    for i, prob_instance in zip(problem_indices, problems):
        code_loc = os.path.join(args.save, f"{args.prefix}{i}.json")
        log_loc = os.path.join(args.save, f"{args.prefix}{i}.log")

        if not args.rerun:
            # if not forcing rerun, check if this experiment has run or failed before
            if os.path.exists(code_loc):
                print(f"Found {code_loc}, args.rerun not enabled, skipping")
                continue
            elif os.path.exists(log_loc):
                print(f"Problem {i} has failed before, args.rerun not enabled, skipping")
                continue

        print(f"Solving Problem #{i}")

        if args.dataset == 'apps':
            from program_env import APPSProgramEnv
            env = APPSProgramEnv(
                prob_path=prob_instance,
                tokenizer=tokenizer,
                model_name=args.load,
                horizon=args.horizon,
                public_test_cases=args.public_cases
            )
        else:
            raise Exception(f"Unknown dataset {args.dataset}")

        # set up models
        dp = APPSHeuristic(
            tokenizer=tokenizer,
            model=model,
            value_model=value_model,
            k=args.width,
            num_beams=args.num_beams,
            test_all_beams=args.test_all_beams,
            horizon=args.horizon,
            new_token_num=args.new_token_num,
            device=args.device,
            use_seq_cache=not args.no_seq_cache,
            use_prompt_cache=not args.no_prompt_cache,
            top_k_cache_steps=args.top_k_cache_steps,
            ts_mode=args.ts_mode,
            env=env,
            debug=args.debug
        )

        start = time.time()

        if args.peek:
            # for sanity check, use the ground truth solution
            states = [env.get_canonical_state()]
            info = {'sample_times': 0}
        else:
            # run code generation
            if args.alg == 'mcts':
                from uct import uct_exp
                states, info = uct_exp(args, env, dp, log_loc, start)
            elif args.alg == 'mcts-multi':
                from uct import uct_multistep_exp
                states, info = uct_multistep_exp(args, env, dp, log_loc, start)
            elif args.alg == 'bs':
                states, info = bs_exp(args, env, dp)
            elif args.alg == 'sample':
                states, info = sample_exp(args, env, dp)
            else:
                raise Exception(f"Unknown alg {args.alg}.")

        if states is None or len(states) == 0:
            continue

        if 'times' in info:
            time_elapsed = info['times']
        else:
            # if time per sample is not available, use the total time
            time_elapsed = time.time() - start

        output_strs = [env.convert_state_to_program(s) for s in states]

        train_rewards = [env.get_reward(s, mode='train') for s in states]
        test_rewards = [env.get_reward(s, mode='test') for s in states]

        best_idx = np.argmax(train_rewards)

        print('final program:')
        print(output_strs[best_idx])
        print('train reward', train_rewards[best_idx])
        print('test reward', test_rewards[best_idx])
        print('time elapsed', time_elapsed[-1] if isinstance(time_elapsed, list) else time_elapsed)
        print('sample times', info['sample_times'])

        with open(code_loc, "w") as f:
            json.dump({'codes': output_strs, 'rewards': test_rewards, 'train rewards': train_rewards,
                       'time': time_elapsed, 'sample times': info['sample_times']}, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="gpt2", choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST)
    parser.add_argument("-l", "--load", default="../models/1.5B", type=str)
    parser.add_argument("--load-value", default=None, type=str, help="An optional value function for evaluating partial programs.")
    parser.add_argument("-t","--test-loc", default="../data_split/test.json", type=str, help="This file specifies the locations of the test set of the code dataset.")
    parser.add_argument("--width", default=3, type=int, help="The maximum number of children for any node.")
    parser.add_argument("--horizon", default=1024, type=int, help="The maximum number of tokens to generate.")
    parser.add_argument("--new-token-num", default=None, type=int, help="The number of new tokens to generate before calling the value function."
                                                                        "None means using the complete horizon (args.horizon).")
    parser.add_argument("--rollout", default=1, type=int, help="The maximum number of rollouts for PG-TD.")
    parser.add_argument("--num-beams", default=1, type=int, help="The number of beams for beam search or PG-TD.")
    parser.add_argument("--num-samples", default=1, type=int, help="The number of samples for Sampling + Filtering.")
    parser.add_argument("--test-all-beams", action='store_true', default=False, help="If True, will run all the beams on test cases to find the best program, which is more time-consuming;"
                                                                                     "otherwise, simply return the most-likely sequence after beam search.")
    parser.add_argument("--ts-mode", default="best", choices=["best", "sample"], help="Tree search mode within the evaluation step. `best` uses beam search, `sample` uses sampling.")

    parser.add_argument("--max-sample-times", default=768, type=int, help="The maximum number of Transformer generation function calls."
                                                                          "Program stops when this number is reached (default to be 512 * 1.5 = 768).")
    parser.add_argument("--time-limit", default=10000, type=int, help="Time limit in sec."
                                                                      "Program stops when time limit is reached.")

    parser.add_argument("--ucb-constant", default=4., type=float)
    parser.add_argument("--ucb-base", default=10., type=float)

    """
    mcts: Planning-Guided Transformer Decoding
    mcts-multi: A multi-step version of mcts, where the agent iteratively performs MCTS and outputs one token at a time, similar to AlphaGo.
    bs: Beam search
    sample: Sample + filtering
    """
    parser.add_argument("--alg", default="mcts", choices=["mcts", "mcts-multi", "bs", "sample"])
    parser.add_argument("--task", default="gen_code", choices=["gen_code", "gen_test"], help="Enable gen_test to output test cases instead of code."
                                                                                             "Only works for HumanEval environment for now.")

    parser.add_argument("--uct-alg", default="var_p_uct", choices=["uct", "p_uct", "var_p_uct"],
                        help="The UCT algorithm to use."
                             "`uct` is the original UCT algorithm,"
                             "`p_uct` is the UCT algorithm with PUCT,"
                             "and `var_p_uct` is the UCT algorithm with variable PUCT.")

    parser.add_argument("--entropy-weighted-strategy", default='none', choices=['none', 'linear', 'linear_with_minimum'])

    parser.add_argument("--peek", action="store_true")

    parser.add_argument("--dataset", default="apps", type=str, choices=["apps"])
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("--indices", default=None, type=str)

    parser.add_argument("--save", type=str, default="./results", help="Directory to save generated code.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix of generated code file.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument('--no-cuda', action='store_true', default=False)

    parser.add_argument('--rerun', action='store_true', default=False, help="If True, rerun if the output file already exists.")
    parser.add_argument('--no-seq-cache', action='store_true', default=False)
    parser.add_argument('--no-prompt-cache', action='store_true', default=False)
    parser.add_argument('--top-k-cache-steps', type=int, default=1024, help="Number of forward steps to cache top k caches, default 1024 means the whole horizon.")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

    # this can be 'desc' for parsing from problem description, 'half' for using half of input_output for public test cases,
    # or a number, that uses the first a few in input_output for public test cases
    parser.add_argument("--public-cases", type=str, default='half', help="Number of public test cases to use for evaluation.")
    parser.add_argument('--overfit', action='store_true', default=False, help="Use the private test case as public tesst case for generation.")
    parser.add_argument('--early-stop', action='store_true', default=False, help="Stop when a program with reward=1 is found.")

    args = parser.parse_args()

    args.device = torch.device('cuda') if torch.cuda.is_available() and not args.no_cuda\
                  else torch.device('cpu')

    if args.alg == 'sample':
        args.ts_mode = 'sample'

    print(pprint.pformat(vars(args)))

    main()