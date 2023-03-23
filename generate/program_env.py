import copy
import json
import os
from abc import abstractmethod, ABC
from collections import OrderedDict
from types import SimpleNamespace

from eval.compute_reward import compute_reward
from transformer_utils.utils import is_codex_model
from eval.generate_gpt_codes import get_output_str_from_state_for_apps


class ProgramEnv(ABC):
    """
    Code generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Reward: pass rate of the program (on the training set in training, and on the test set in testing).
    """
    def __init__(self, terminal_token, horizon):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_token = terminal_token
        self.horizon = horizon

        # state -> reward
        # we may need to retrieve the states (programs) in the order they were saved, so use OrderedDict
        self.cached_reward = OrderedDict()

    def transition(self, s, a, is_model_dynamic=True):
        next_state = s + [a]

        if a == self.terminal_token or len(next_state) == self.horizon:
            # either the program finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            reward = self.get_reward(next_state)
        else:
            reward = 0  # no intermediate reward

        return next_state, reward, done

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)

        return self.state, reward, done, {}

    @abstractmethod
    def get_reward(self, s, mode='train'):
        """
        This needs to be defined for each dataset
        """
        pass

    def convert_state_to_program(self, s):
        """
        The state may be different from the program. This converts it back to executable program.
        """
        return s

    def equality_operator(self, s1, s2):
        return s1 == s2

    def get_complete_programs(self):
        """
        Return the list of complete programs reached so far.
        This can be found from the list of cached rewards.
        """
        return list(map(lambda x: list(x), self.cached_reward.keys()))


class APPSProgramEnv(ProgramEnv):
    """
    Code generation environment for APPS dataset.
    """
    def __init__(self, prob_path, tokenizer, model_name, horizon, public_test_cases=None):
        self.prob_path = prob_path
        self.tokenizer = tokenizer
        self.model = model_name
        self.public_test_cases = public_test_cases

        # code from generate_gpt_codes that generate paths for all essential files
        public_test_case_path = os.path.join(prob_path, "public_input_output.json")
        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")

        if not os.path.exists(starter_path):
            starter_path = None
        if not os.path.exists(test_case_path):
            raise Exception("input_output.json missing so can't do testing. Invalid ProgramEnv.")
        if not os.path.exists(prompt_path):
            raise Exception("question.json missing. Invalid ProgramEnv.")
        if public_test_cases == 'desc' and not os.path.exists(public_test_case_path):
            raise Exception('using public test cases in problem description, but public test cases missing.')

        if is_codex_model(self.model):
            from transformer_utils.codex import generate_apps_prompt_for_codex
            # add `Python 3` to the prompt
            state, _ = generate_apps_prompt_for_codex(prompt_path)
        else:
            from eval.generate_gpt_codes import generate_apps_prompt
            # generate prompt to encode question text and an "ANSWER" prompt to the state
            # no need to load the full arglist here, it only needs to check the value of peeking (using default value 0.0 here)
            gpt_args = SimpleNamespace(peeking=0.0)
            state, _ = generate_apps_prompt(gpt_args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)

        self.init_prompt = copy.copy(state)

        self.state = self.tokenizer.encode(state)
        if is_codex_model(self.model):
            terminal_token = '<|endoftext|>'
        else:
            terminal_token = self.tokenizer.encode('<|endoftext|>')[0]

        super(APPSProgramEnv, self).__init__(terminal_token=terminal_token, horizon=horizon)

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        if is_codex_model(self.model):
            # Codex likes to write a main function, the testing environment is not '__main__'
            # replace the if statements here
            s = s.replace("if __name__ == '__main__'", 'if True')
            s = s.replace("if __name__ == \"__main__\"", 'if True')

            # get rid of prompt
            return s[len(self.init_prompt):]
        else:
            return get_output_str_from_state_for_apps(s)

    def get_canonical_state(self):
        raise NotImplementedError()

    def get_reward(self, s, mode='train'):
        """
        Returns:
            The reward of program in s.
        """
        if s is None:
            return 0

        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            return self.cached_reward[tuple(s)]

        output_str = self.convert_state_to_program(s)
        reward = compute_reward(self.prob_path, output_str, mode=mode, public_test_cases=self.public_test_cases)

        if mode == 'train':
            self.cached_reward[tuple(s)] = reward

        return reward


class HumanEvalProgramEnv(ProgramEnv):
    def __init__(self, problem, model_name, tokenizer, horizon, generated_test_cases=None, metric=None, task='gen_code', peek=False, overfit=False):
        """
        Args:
            problem: a problem object in HumanEval
            generated_test_cases: should be generated using task='gen_test' first
            peek: use ground truth solution, a way to check if generated test cases are good
        """
        self.problem = problem
        self.tokenizer = tokenizer
        self.model = model_name
        if task == 'gen_code':
            assert generated_test_cases is not None, "require generated_test_cases for generation task"
            with open(generated_test_cases, 'r') as f:
                json_data = json.load(f)
                self.generated_test_cases = json_data['code']
        self.metric = metric
        self.task = task
        self.peek = peek
        self.overfit = overfit

        # define initial prompt
        state = self.problem['prompt']

        if self.task == 'gen_test':
            state = generate_human_eval_test_prompt(state, self.problem['entry_point'])

        self.init_prompt = copy.copy(state)
        self.state = self.tokenizer.encode(state)

        self.test = self.problem['test']

        if is_codex_model(self.model):
            terminal_token = '<|endoftext|>'
        else:
            raise Exception("APPS model cannot run on HumanEval.")

        super(HumanEvalProgramEnv, self).__init__(terminal_token=terminal_token, horizon=horizon)

    def get_reward(self, s, mode='train'):
        if self.task == 'gen_test':
            # test case generation does not need rewards
            return 0

        if mode == 'train' and tuple(s) in self.cached_reward.keys():
            # use cached rewards for training
            return self.cached_reward[tuple(s)]
        program = self.convert_state_to_program(s)

        if mode == 'train' and not self.overfit:
            reference = self.generated_test_cases
        else:
            reference = self.test # use ground truth

        # need to call the check function
        reference = f"{reference}\ncheck({self.problem['entry_point']})"

        pass_at_k, results = self.metric.compute(references=[reference], predictions=[[program]], k=[1])
        print(pass_at_k)
        print(results)
        reward = pass_at_k['pass@1']

        if mode == 'train':
            self.cached_reward[tuple(s)] = reward

        return reward

    def get_canonical_state(self):
        """
        Returns: the solution found in the dataset
        """
        program = self.init_prompt + self.problem['canonical_solution']
        return self.tokenizer.encode(program)

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        if is_codex_model(self.model):
            if self.task == "gen_test":
                # need to get the codes that do the assertions
                # it's highly likely that some assertions did not finish. trim the last incomplete line in this case
                assertions = s[len(self.init_prompt) - 42:]
                if assertions[-1] != '\n':
                    last_slash_n = assertions.rfind('\n')
                    if last_slash_n != -1:
                        return assertions[:last_slash_n+1]
                return assertions
            else:
                return s
        else:
            raise Exception("unsupported model")


def generate_human_eval_test_prompt(state, func_name):
    lines = state.split('\n')

    # remove example inputs, outputs
    comment_symbol_counter = 0
    clip_start_line = None
    clip_end_line = None
    for line_num, line in enumerate(lines):
        if '   """' in line:
            comment_symbol_counter += 1
            if comment_symbol_counter == 2:
                clip_end_line = line_num
        elif '    >>>' in line and clip_start_line is None:
            clip_start_line = line_num

    clipped_lines = lines[:clip_start_line] + lines[clip_end_line:]
    state = '\n'.join(clipped_lines)

    # modify the prompt for test case generation according to
    # CodeT: Code Generation with Generated Tests  https://arxiv.org/abs/2207.10397
    state += f"""   pass

    # check the correctness of {func_name}
    def check(candidate):
        assert candidate"""

    return state