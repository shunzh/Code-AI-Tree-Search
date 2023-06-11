import copy
import json
import os
import warnings
from abc import abstractmethod, ABC
from collections import OrderedDict
from types import SimpleNamespace

from eval.compute_reward import compute_reward
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

        from eval.generate_gpt_codes import generate_apps_prompt
        # generate prompt to encode question text and an "ANSWER" prompt to the state
        # no need to load the full arglist here, it only needs to check the value of peeking (using default value 0.0 here)
        gpt_args = SimpleNamespace(peeking=0.0)
        state, _ = generate_apps_prompt(gpt_args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)

        self.init_prompt = copy.copy(state)

        self.state = self.tokenizer.encode(state)
        terminal_token = self.tokenizer.encode('<|endoftext|>')[0]

        super(APPSProgramEnv, self).__init__(terminal_token=terminal_token, horizon=horizon)

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
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
