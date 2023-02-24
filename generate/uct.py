import time
import traceback

from scipy.stats import entropy

from default_pi import DefaultPolicyHeuristic
from dyna_gym.agents.mcts import plot_tree, update_root, DecisionNode, pre_order_traverse
from dyna_gym.agents.uct import UCT
from eval.utils import log_error


def uct_exp(args, env, dp, log_loc, start):
    """
    Run TG-MCTS
    """
    agent = UCT(
        action_space=[], # this will not be used as we have a default policy
        gamma=1., # no discounting
        ucb_constant=args.ucb_constant,
        ucb_base=args.ucb_base,
        horizon=args.horizon,
        rollouts=args.rollout,
        dp=dp,
        width=args.width,
        ts_mode=args.ts_mode,
        reuse_tree=True,
        alg=args.uct_alg
    )

    agent.display()

    # tell the mcts to stop when this is satisfied
    term_cond = lambda: dp.sample_times > args.max_sample_times or time.time() - start > args.time_limit \
                        or (args.early_stop and max(env.cached_rewards.values()) == 1.)

    try:
        done = False
        s = env.state

        if len(s) >= args.horizon:
            log_error(f'Cannot process programs longer than {args.horizon}. Stop here.', log_loc)
            return None, None
        else:
            # run mcts. a bit hacky, but this will run mcts simulation. we do not need to take any action
            agent.act(env, done, term_cond=term_cond)
    except Exception as e:
        if args.debug:
            raise e
        else:
            print("Unexpected exception in generating solution")
            log_error(traceback.format_exc() + '\n', log_loc)
            return None, None

    # these may not be assigned, set default values
    if args.debug:
        # print the mcts tree
        try:
            plot_tree(agent.root, env, log_loc)
        except Exception as e:
            print(f"Error plotting tree.\n{e}")
            print(traceback.format_exc())

    states = env.get_complete_programs()

    time_stamps = dp.time_stamps
    times = [t - start for t in time_stamps]

    return states, {'sample_times': dp.sample_times, 'times': times}


def uct_multistep_exp(args, env, dp, log_loc, start):
    agent = UCT(
        action_space=[],  # this will not be used as we have a default policy
        gamma=1.,  # no discounting
        ucb_constant=args.ucb_constant,
        ucb_base=args.ucb_base,
        horizon=args.horizon,
        rollouts=args.rollout,
        dp=dp,
        width=args.width,
        reuse_tree=True,
        alg=args.uct_alg,
    )

    agent.display()

    try:
        done = False
        s = env.state
        for t in range(args.horizon):
            if dp.sample_times > args.max_sample_times:
                print('Maximum number of samples reached.')
                break

            if time.time() - start > args.time_limit:
                print('Time exceeded.')
                break

            if len(s) >= args.horizon:
                print(f'Cannot process programs longer than {args.horizon}. Stop here.')
                break

            if done:
                break

            if t > 0:
                # tree is not built at time step 0 yet
                ent = entropy([child.prob for child in agent.root.children])
            else:
                ent = 1  # this wouldn't change the rollout number

            if args.entropy_weighted_strategy == 'linear':
                rollout_weight = ent
            elif args.entropy_weighted_strategy == 'linear_with_minimum':
                rollout_weight = 0.2 + 0.8 * ent
            elif args.entropy_weighted_strategy == 'none':
                rollout_weight = 1  # does not change rollout number
            else:
                raise ValueError(f'Unknown rollout strategy {args.entropy_rollout_strategy}')

            act = agent.act(env, done, rollout_weight=rollout_weight)
            s, r, done, _ = env.step(act)

            if args.debug:
                # print the current tree
                print('tree:')
                plot_tree(agent.root, env, log_loc)

                print('took action:')
                act_str = env.tokenizer.decode(act)
                print(repr(act_str))
                print('========== state (excluding prompt) ==========')
                print(env.convert_state_to_program(s))

                print('entropy at this step: ', ent)

            update_root(agent, act, s)
            dp.clean_up(s)
    except Exception as e:
        if args.debug:
            raise e
        else:
            print("Unexpected exception in generating solution")
            log_error(traceback.format_exc() + '\n', log_loc)
            return None, None

    if len(s) >= args.horizon:
        log_error('Exceeds horizon.\n', log_loc)
        return None, None

    states = env.get_complete_programs()

    time_stamps = dp.time_stamps
    times = [t - start for t in time_stamps]

    return states, {'sample_times': dp.sample_times, 'times': times}
