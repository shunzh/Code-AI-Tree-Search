"""
This script parses the public test cases from the problem descriptions for APPS.
Our experiments results did not use these for evaluation, however.
"""
import json
import os
import re
import warnings

possible_prompts = ['Sample Input:', 'Sample Input [0-9]*:', 'Input',
                    'Sample Output:', 'Sample Output [0-9]*:', 'Output',]

def extract_public_test_case_from_description(problem: str, desc: str):
    if '-----Example' in desc:
        if '-----Examples-----' in desc:
            desc = desc.split('-----Examples-----\n')[1]
        elif '-----Example-----' in desc:
            desc = desc.split('-----Example-----\n')[1]
        else:
            warnings.warn(f"failed to cut at 'Example' for {problem}.")
            return None

        in_outs = re.split('|'.join([prompt + '\n' for prompt in possible_prompts]), desc)
    elif '-----Sample Input-----' in desc and '-----Sample Output-----' in desc:
        in_outs = re.split('-----Sample Input-----\n|-----Sample Output-----\n', desc)
    else:
        warnings.warn(f"cannot process description for {problem}.")
        return None

    # the test cases are after the first prompt
    in_outs = in_outs[1:]

    if len(in_outs) == 0:
        warnings.warn(f"empty in_out for {problem}.")
        return None

    # get rid of text after the last output
    in_outs[-1] = in_outs[-1].split('\n\n')[0]
    in_outs[-1] += '\n'

    in_outs = [in_out.replace('\n\n', '\n') for in_out in in_outs]

    inputs = in_outs[0::2]
    outputs = in_outs[1::2]
    return {'inputs': inputs, 'outputs': outputs}

if __name__ == '__main__':
    test_loc = "../data_split/test.json"

    with open(test_loc, "r") as f:
        problems = json.load(f)
    problems = sorted(problems) # Pin some ordering

    for problem in problems:
        prompt_path = os.path.join(problem, "question.txt")
        with open(prompt_path, "r") as f:
            question = f.read()
            in_outs = extract_public_test_case_from_description(problem, question)

            if in_outs is not None:
                public_test_case_loc = os.path.join(problem, "public_input_output.json")
                with open(public_test_case_loc, 'w') as f:
                    json.dump(in_outs, f)