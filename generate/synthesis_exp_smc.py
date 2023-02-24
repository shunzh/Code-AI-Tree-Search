import json
import os
import pprint
import sys
import time
import traceback

import torch
import transformers
import numpy as np

sys.path.append('../')
sys.path.append('../eval/')

from program_env import APPSProgramEnv

# okay with parallelization
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log_error(msg:str, log_loc:str):
    print(msg)
    with open(log_loc, 'w') as f:
        f.write(msg)

def exp(args, problem_indices):
    """
    Args:
        problem_indices: the indices of the programs that we're going to solve

    Returns:
        None
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.arch)

    print(f"Loading model {args.load}")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.load, pad_token_id=tokenizer.eos_token_id)
    model.to(args.device)
    if args.device == torch.device('cuda') and hasattr(model, 'parallelize'):
        model.parallelize()
    print("Model loaded.")

    # get problem locations
    with open(args.test_loc, "r") as f:
        problems = json.load(f)
    # get the programs requested to run
    problems = [problems[idx] for idx in problem_indices]

    for i, prob_path in zip(problem_indices, problems):
        code_loc = os.path.join(args.save, f"{args.prefix}{i}.json")
        log_loc = os.path.join(args.save, f"{args.prefix}{i}.log")

        if not args.rerun:
            # if not forcing rerun, check if this experiment has run before, or failed before
            if os.path.exists(code_loc):
                print(f"Problem {i} already generated and args.rerun not enabled, skipping")
                continue
            elif os.path.exists(log_loc):
                print(f"Problem {i} has failed before and args.rerun not enabled, skipping")
                continue

        print(f"Solving Problem {prob_path}")

        env = APPSProgramEnv(
            prob_path=prob_path,
            tokenizer=tokenizer,
            model_name=args.load,
            horizon=args.horizon,
            public_test_cases=args.public_cases
        )

        # for fair comparison, loading models and tokenizers are not included in computation time
        start = time.time()

        states = [env.state] * args.pop_size
        fitness = [1. / args.pop_size] * args.pop_size

        output_states = []

        sample_times = 0

        t = 0
        output_hash = []

        try:
            while t < args.horizon and len(states) > 0:
                print('time step', t)
                if any(len(s) >= args.horizon for s in states):
                    print(f'Cannot process programs longer than {args.horizon}. Stop here.')
                    break

                if time.time() - start > args.time_limit:
                    print('Time exceeded.')
                    break

                new_states = []
                for next_state_id in range(len(states)):
                    state_idx = np.random.choice(range(len(states)), p=fitness)
                    s = states[state_idx]

                    input_ids = torch.LongTensor(s).unsqueeze(0).to(args.device)
                    output_ids = model.generate(
                        input_ids,
                        top_k=args.k,
                        early_stopping=True,
                        max_new_tokens=1,
                        do_sample=True,
                        use_cache=True
                    )

                    next_state = output_ids[0].tolist()
                    if next_state[-1] != env.terminal_token:
                        new_states.append(next_state)
                    else:
                        output_states.append(next_state)

                if len(output_states) > 0: print('output states num', len(output_states))

                fitness = []
                for s in new_states:
                    output_ids = None
                    if args.num_beams == 1 and args.use_seq_cache:
                        # If no beam search is used, if the prefix of a previously generated sequences generated state matches
                        # state, Transformer will generate the exact sequence. So use cache.
                        for cached_ids in output_hash:
                            if s == cached_ids[:len(s)]:
                                if args.debug: print('sequence cache hit')
                                output_ids = cached_ids
                                break

                    if output_ids is None:
                        # fail to obtain from cache
                        input_ids = torch.LongTensor(s).unsqueeze(0).to(args.device)

                        output_ids = model.generate(
                            input_ids,
                            top_k=args.k,
                            num_beams=args.num_beams,
                            early_stopping=True,
                            max_length=args.horizon,
                            use_cache=True # huggingface default cache is always enabled
                        )

                        output_ids = output_ids[0].tolist()
                        # save to cache
                        if args.use_seq_cache:
                            output_hash.append(output_ids)

                        sample_times += 1

                    fitness.append(env.get_reward(output_ids))


                print('fitness', fitness)
                if args.debug and len(new_states) > 0:
                    print(env.convert_state_to_program(new_states[0]))

                # normalize
                if len(fitness) > 0:
                    if np.sum(fitness) == 0:
                        # reset fitness to uniform
                        fitness = [1. / len(fitness)] * len(fitness)
                    else:
                        fitness /= np.sum(fitness)

                states = new_states
                t += 1
        except Exception as e:
            print("Unexpected exception in generating solution")
            print(e)

            # write errors to log file
            log_error(traceback.format_exc() + '\n', log_loc)
            continue

        if len(output_states) == 0:
            log_error('No complete programs are generated within time budget.\n', log_loc)
            # exceeding horizon, count it as failure
            continue

        time_elapsed = time.time() - start

        all_states = env.get_complete_programs()
        output_strs = [env.convert_state_to_program(s) for s in all_states]
        train_rewards = [env.get_reward(s, mode='train') for s in all_states]
        test_rewards = [env.get_reward(s, mode='test') for s in all_states]

        best_idx = np.argmax(train_rewards)

        print('final program:')
        print(output_strs[best_idx])
        print('train reward', train_rewards[best_idx])
        print('test reward', test_rewards[best_idx])
        print('time elapsed', time_elapsed)

        if not os.path.exists(args.save):
            os.makedirs(args.save, exist_ok=True)

        with open(code_loc, "w") as f:
            json.dump({'codes': output_strs, 'rewards': test_rewards, 'train rewards': train_rewards,
                       'time': time_elapsed, 'sample times': sample_times}, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="gpt2", choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST)
    parser.add_argument("-l", "--load", default="../models/1.5B", type=str)
    parser.add_argument("-t","--test-loc", default="../data_split/test.json", type=str)
    parser.add_argument("--horizon", default=1024, type=int)
    parser.add_argument("--num-beams", default=1, type=int)
    parser.add_argument("--pop-size", default=10, type=int, help="Population size.")
    parser.add_argument("-k", default=3, type=int) # suggested by reviewer, use consistent k
    parser.add_argument("--time-limit", default=10000, type=int, help="Time limit in sec.")

    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("--indices", default=None, type=str)

    parser.add_argument("--save", type=str, default="./results")
    parser.add_argument("--prefix", type=str, default="", help="Prefix of generated code file.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument('--no-cuda', action='store_true', default=False)

    parser.add_argument('--rerun', action='store_true', default=False)
    parser.add_argument('--no-seq-cache', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

    parser.add_argument("--public-cases", type=str, default='half')

    args = parser.parse_args()

    args.device = torch.device('cuda') if torch.cuda.is_available() and not args.no_cuda\
                  else torch.device('cpu')

    args.use_seq_cache = not args.no_seq_cache

    print(pprint.pformat(vars(args)))

    if args.index is not None:
        problem_indices = [args.index]
        log_id = args.index
    elif args.end is not None:
        problem_indices = range(args.start, args.end)
        log_id = str(problem_indices)
    elif args.indices is not None:
        # customized list (maybe cases that have high performances under Transformer)
        with open(args.indices) as f:
            problem_indices = json.load(f)
        log_id = args.indices
    else:
        raise Exception("Don't know what problems to solve.")


    exp(args, problem_indices)