import json
import os


def get_test_cases(root, mode, public_test_cases, debug=False):
    if public_test_cases == 'desc':
        # use test cases in problem description as public test cases
        if mode == 'train':
            io_file = "public_input_output.json"
        elif mode == 'test':
            io_file = "input_output.json"
        else:
            raise Exception(f"Unknown mode {mode}")

        if os.path.exists(os.path.join(root, io_file)):
            with open(os.path.join(root, io_file)) as f:
                in_outs = json.load(f)
                if debug:
                    print(f"test cases json = {in_outs['inputs']} {in_outs['outputs']}")
        else:
            in_outs = None
    else:
        if os.path.exists(os.path.join(root, "input_output.json")):
            with open(os.path.join(root, "input_output.json")) as f:
                in_outs = json.load(f)

                if mode != 'all' and public_test_cases != 'all':
                    # if 'all' is specified, simply use all in_outs without splitting
                    if 'train_set_size' in in_outs.keys() and 'test_set_size' in in_outs.keys():
                        # in this case, the dataset specified which are the train and test data
                        train_set_size = in_outs['train_set_size']
                        test_set_size = in_outs['test_set_size']

                        if train_set_size == 0 or test_set_size == 0:
                            raise Exception(f"Empty training or test sets.")

                        if mode == 'train':
                            in_outs['inputs'] = in_outs['inputs'][0:train_set_size]
                            in_outs['outputs'] = in_outs['outputs'][0:train_set_size]
                        elif mode == 'test':
                            in_outs['inputs'] = in_outs['inputs'][train_set_size:train_set_size + test_set_size]
                            in_outs['outputs'] = in_outs['outputs'][train_set_size:train_set_size + test_set_size]
                    else:
                        # if train, test split is not provided in the dataset, split in half
                        in_out_len = len(in_outs['inputs'])

                        if public_test_cases == 'half':
                            # split evenly by default
                            public_test_cases = in_out_len // 2
                        elif public_test_cases.isdigit():
                            public_test_cases = int(public_test_cases)
                        else:
                            raise Exception(f"Can't understand public_test_cases {public_test_cases}")
                        private_test_cases = in_out_len - public_test_cases

                        if public_test_cases < 1 or private_test_cases < 1:
                            raise Exception(f"Not enough test cases: {public_test_cases}, {private_test_cases}.")

                        if mode == 'train':
                            in_outs['inputs'] = in_outs['inputs'][:public_test_cases]
                            in_outs['outputs'] = in_outs['outputs'][:public_test_cases]
                        elif mode == 'test':
                            in_outs['inputs'] = in_outs['inputs'][public_test_cases:]
                            in_outs['outputs'] = in_outs['outputs'][public_test_cases:]
        else:
            in_outs = None # will raise an Exception later

    return in_outs
