"""
The original code generation script from APPS. We no longer use this script for our experiments.
"""

import io
import json
import random
import traceback

import numpy as np
import os
import pprint
import time
import transformers
import torch

from compute_reward import compute_reward
from transformer_utils.utils import is_codex_model
from reindent import run as run_reindent

# for timing and debugging
from tqdm import tqdm

# okay with parallelization
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def generate_apps_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"

    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # Need to do some peeking.

        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol

def get_output_str_from_state_for_apps(s):
    """
    Get the code from the transformer output
    """
    if "ANSWER:" in s:
        s = s.split("ANSWER:\n")[1]

    return s.replace("<|endoftext|>", "")

def log_error(msg:str, log_loc:str):
    print(msg)
    with open(log_loc, 'w') as f:
        f.write(msg)

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    with open(args.test_loc, "r") as f:
        problems = json.load(f)
    problems = sorted(problems) # Pin some ordering

    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)

    # Only do the problems that are specified.
    if args.index is not None:
        problems = [problems[args.index]]
        indices = [args.index]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]
        indices = range(start, end)

    # Set up model
    print("Loading model...")
    if is_codex_model(args.load):
        # this is a codex model
        tokenizer = None # no tokenizer needed or provided

        from transformer_utils.codex import codex_generate, generate_prompt_for_codex

        generate_prompt = generate_prompt_for_codex
        get_output_str_from_state = lambda s: s # since problem description is in the comment block, the generated code is directly runnable
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.arch)

        model = transformers.AutoModelForCausalLM.from_pretrained(args.load, pad_token_id=tokenizer.eos_token_id)
        model.to(args.device)

        generate_prompt = generate_apps_prompt
        get_output_str_from_state = get_output_str_from_state_for_apps

        if args.device == torch.device('cuda') and hasattr(model, 'parallelize'):
            model.parallelize()
    print(f"Loaded/initialized {args.load}.")

    # main eval loop
    for index, problem in zip(indices, tqdm(problems)):
        code_loc = os.path.join(args.save, f"{args.prefix}{index}.json")
        log_loc = os.path.join(args.save, f"{args.prefix}{index}.log")

        if not args.rerun:
            if os.path.exists(code_loc):
                # this code is already generated, and not required to regenerate
                print(f"skip {index} because code exists")
                continue
            elif os.path.exists(log_loc):
                print(f"skip {index} because it failed before (log file found)")
                continue

        prob_path = os.path.join(args.root, problem)
        if args.debug:
            print(f"problem path = {prob_path}")

        test_case_path = os.path.join(prob_path, "input_output.json")
        public_test_case_path = os.path.join(prob_path, "public_input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
            starter_path = None
        if not os.path.exists(prompt_path):
            log_error('question description missing.', log_loc)
            continue
        if not os.path.exists(test_case_path):
            log_error('test cases missing.', log_loc)
            continue

        # Read the question in
        prompt_text, sample_sol = generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)
        if args.debug:
            print("PROMPT_TEXT:")
            print(prompt_text)
        
        # Feed this into the model.
        start = time.time()
        with torch.no_grad():
            if is_codex_model(args.load):
                if args.sample:
                    output_strs = []

                    for sample_num in tqdm(range(args.sample_size)):
                        output_str, info = codex_generate(input_str=prompt_text, model=args.load, temperature=0.9)
                        if output_str is not None:
                            output_strs.append(output_str)
                else:
                    output_str, info = codex_generate(input_str=prompt_text, model=args.load)

                    if output_str is None:
                        # generation failed
                        log_error(str(info), log_loc)
                        continue

                    output_strs = [output_str]
            else:
                # a gpt model
                input_ids = torch.LongTensor(tokenizer.encode(prompt_text, verbose=False)).unsqueeze(0).to(args.device)
                if len(input_ids[-1]) >= 1024:
                    log_error('prompt length exceeds 1024, skip.', log_loc)
                    continue

                if args.sample:
                    # CUDA may not have enough memory to sample in one batch
                    # compute a schedule to decide how many output_ids to generate in each batch
                    sample_schedule = [args.sample_batch_size] * (args.sample_size // args.sample_batch_size)
                    sample_residue = args.sample_size - args.sample_batch_size * len(sample_schedule)
                    if sample_residue > 0:
                        sample_schedule.append(sample_residue)

                    output_ids_list = []
                    # show progress bar if generating in multiple batches
                    if len(sample_schedule) > 1: sample_schedule = tqdm(sample_schedule)
                    for sample_num in sample_schedule:
                        try:
                            output_ids = model.generate(
                                input_ids,
                                #num_beams=args.num_beams, # this should be disabled for sampling, otherwise running beam sample
                                early_stopping=True,
                                top_k=args.k,
                                top_p=args.p,
                                num_return_sequences=sample_num,
                                do_sample=args.sample,
                                max_length=1024,
                                temperature=args.temperature
                            )
                            output_ids_list.append(output_ids)

                        except Exception as e:
                            if isinstance(e, UnboundLocalError) and str(e) == "local variable 'next_tokens' referenced before assignment":
                                # See https://github.com/huggingface/transformers/issues/5118
                                error = "Problem text was > 1024 tokens, so cannot do generation"
                            else:
                                error = "Unexpected exception in generating solution\n"
                                error += traceback.format_exc() + '\n'

                            log_error(error, log_loc)
                            # proceed to next SAMPLE
                            continue

                    output_strs = [tokenizer.decode(output)
                                   for output_ids in output_ids_list
                                   for output in output_ids]
                else:
                    try:
                        # make sure using the exact function call as APPS
                        # max_length was 1024 - len(input_ids). seems to be a bug, fixed
                        output_ids = model.generate(
                            input_ids,
                            num_beams=args.num_beams,
                            early_stopping=True,
                            max_length=1024
                        )
                    except Exception as e:
                        if isinstance(e, UnboundLocalError) and str(e) == "local variable 'next_tokens' referenced before assignment":
                            # See https://github.com/huggingface/transformers/issues/5118
                            error = "Problem text was > 1024 tokens, so cannot do generation"
                        else:
                            error = "Unexpected exception in generating solution\n"
                            error += traceback.format_exc() + '\n'

                        log_error(error, log_loc)
                        # proceed to next problem
                        continue

                    output_strs = [tokenizer.decode(output_ids[0])]

        time_elapsed = time.time() - start

        if args.peeking == 1.0:
            output_strs = [sample_sol]
            train_rewards, test_rewards = [], []
        elif len(output_strs[0]):
            output_strs = [get_output_str_from_state(output) for output in output_strs]

            if len(output_strs) > 1:
                train_rewards = [compute_reward(prob_path, output_str, mode='train', public_test_cases=args.public_cases)
                                 for output_str in output_strs]
            else:
                train_rewards = []

            test_rewards = [compute_reward(prob_path, output_str, mode='test', public_test_cases=args.public_cases)
                            for output_str in output_strs]
        else:
            # program fails to be generated, continue
            continue

        if args.debug:
            print(f"Generation time: {time_elapsed}")
            print(f"Generated output string:")
            print(output_strs[0])
            print("------------------------------------------------------------")

        with open(code_loc, "w") as f:
            json.dump({'codes': output_strs, 'rewards': test_rewards, 'train rewards': train_rewards,
                       'time': time_elapsed, 'sample times': args.sample_size}, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a tranined model to generate Python code.")
    parser.add_argument("--arch", default="gpt2", choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST)
    parser.add_argument("-t","--test-loc", default="../data_split/test.json", type=str)
    parser.add_argument("-r","--root", default="", type=str, help="where the data is stored.")
    parser.add_argument("-l","--load", default="../models/1.5B", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("--sample-size", default=1, type=int, help='use test cases to evaluate top-k cases')
    parser.add_argument("--sample", action='store_true', help='enable sampling for Transformer')
    parser.add_argument("--sample-batch-size", default=1, type=int, help='number of outputs to generate by Transformer in one batch by sampling')
    parser.add_argument("--temperature", default=1., type=float, help='temperature used for Transformer sampling')

    parser.add_argument("-k", default=3, type=int, help='top-k sampling')
    parser.add_argument("-p", default=None, type=float, help='top-p sampling')

    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results")
    parser.add_argument("--prefix", type=str, default="", help="Prefix of generated code file.")
    # this can be 'desc' for parsing from problem description, 'half' for using half of input_output for public test cases, 'all' for using all of them,
    # or a number, that uses the first a few in input_output for public test cases
    parser.add_argument("--public-cases", type=str, default='half')
    parser.add_argument("--need-training-rewards", action='store_true', default=False)

    parser.add_argument('--rerun', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

    args = parser.parse_args()

    args.device = torch.device('cuda') if torch.cuda.is_available() and not args.no_cuda\
                  else torch.device('cpu')

    main(args)

