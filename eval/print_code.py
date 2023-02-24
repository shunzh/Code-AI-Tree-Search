"""
This script prints the code nicely from the result json file.
"""
import json
import sys


if __name__ == '__main__':
    filename = sys.argv[1]

    code_info = json.load(open(filename))

    if isinstance(code_info, dict):
        print('CODE')
        codes = code_info['codes']

        for code in codes:
            print(code.replace('\\n', '\n').replace('\\t', '\t'))
            print()

        print('COMPUTATION TIME')
        print(code_info['time'])
    elif isinstance(code_info, str):
        print('CODE')
        print(code_info)
    else:
        raise Exception(f"can't print {filename}. Content:\n{code_info}")