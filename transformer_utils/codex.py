import json
import os

import openai
from atomicwrites import atomic_write

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from transformer_utils.base import WrappedTransformer

# make sure read API key from the openai_api_key file in the same directory
api_key_file = os.path.join(os.path.dirname(__file__), 'openai_api_key')
openai.api_key = open(api_key_file, 'r').read().strip()

# since requesting openai codex is slow and subject to a subject, all requestes are cached locally
codex_cache_file = os.path.join(os.path.dirname(__file__), 'codex_cache.json')


def generate_apps_prompt_for_codex(prompt_path):
    with open(prompt_path, "r") as f:
        data = f.read()

    return f'"""\nPython 3\n{data}\n"""\n', None


class CodexModel(WrappedTransformer):
    def __init__(self, model):
        """
        Initialize the cache dictionary if it's not initialized
        """
        self.model = model
        self.codex_request_counter = 0

        if os.path.exists(codex_cache_file):
            print("load from codex cache file.")
            with open(codex_cache_file, 'r') as f:
                self.codex_cache = json.load(f)
        else:
            print("create an empty codex cache dictionary.")
            self.codex_cache = dict()

    # request to codex may fail (mostly because of frequency), this decorates the function to automatically retry sending requests
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def generate(self, input_str, horizon=1024, k=5, temperature=0):
        """
        Args:
            input_str: prompt, str or list of tokens (a state in tree search), both okay
            horizon: longest output length
            k: top-k beam search / sampling
            temperature: 0 is greedily optimal (arg-max sampling); > 0 is sampling

        Returns:
            generated sequence as a str, openai response object
        """
        if isinstance(input_str, list):
            input_str_key = "".join(input_str)
        else:
            input_str_key = input_str

        input_key = input_str_key + '|' + self.model + '|' + str(horizon) + '|' + str(k) # cache key

        if temperature == 0 and input_key in self.codex_cache.keys():
            print("codex cache hit")
            response = self.codex_cache[input_key]
            x = response['choices']
            return x[0]['text'], response
        else:
            if isinstance(input_str, list):
                input_str = [input_str] # need to add the batch dimension

            response = openai.Completion.create(
                model=self.model,
                prompt=input_str,
                temperature=temperature, # output the optimal sequence
                echo=True, # return should include the input_str, so that it can complete partial codes
                logprobs=k,
                max_tokens=horizon,
            )

            if 'choices' in response:
                x = response['choices']
                if len(x) > 0:
                    self.codex_request_counter += 1

                    if temperature == 0:
                        # only save to cache when doing greedy decoding
                        self.codex_cache[input_key] = response

                        if self.codex_request_counter % 50 == 0:
                            if len(self.codex_cache) > 1000:
                                keys_to_del = list(self.codex_cache.keys())[:len(self.codex_cache) - 1000]
                                for key in keys_to_del:
                                    del self.codex_cache[key]

                            with atomic_write(codex_cache_file, overwrite=True) as f:
                                json.dump(self.codex_cache, f)

                    return x[0]['text'], response

            raise Exception("codex return invalid.")

class CodexTokenizer:
    """
    An almost dummy tokenzier. Codex does encoding / decoding in the server,
    so encoding simply convert string to a list of string tokens
    """
    def __init__(self, codex_model):
        self.codex_model = codex_model

    def encode(self, ids):
        if isinstance(ids, str):
            seq, info = self.codex_model.generate(input_str=ids, horizon=0)
            return info['choices'][0]['logprobs']['tokens']
        else:
            assert isinstance(ids, list)
            return ids

    def decode(self, ids):
        if isinstance(ids, str):
            return ids
        else:
            assert isinstance(ids, list)
            s = "".join(ids)

            terminal_idx = s.find('<|endoftext|>')
            if terminal_idx != -1:
                s = s[:terminal_idx]

            return s


if __name__ == '__main__':
    # for sanity check
    test_input = '''
"""
A + B is often used as an example of the easiest problem possible to show some contest platform. However, some scientists have observed that sometimes this problem is not so easy to get accepted. Want to try?


-----Input-----

The input contains two integers a and b (0 ≤ a, b ≤ 10^3), separated by a single space.


-----Output-----

Output the sum of the given integers.


-----Examples-----
Input
5 14

Output
19

Input
381 492

Output
873
"""
'''

    model = CodexModel(model="code-davinci-002")
    model.generate(input_str=test_input)