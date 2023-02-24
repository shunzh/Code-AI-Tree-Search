"""
MCTS Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""
import random
import itertools
import warnings

from tqdm import tqdm

import dyna_gym.utils.utils as utils
from gym import spaces
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from eval.utils import hierarchy_pos


def chance_node_value(node, mode="best"):
    """
    Value of a chance node
    """
    if len(node.sampled_returns) == 0:
        return 0

    if mode == "best":
        # max return (reasonable because the model is deterministic?)
        return max(node.sampled_returns)
    elif mode == "sample":
        # Use average return
        return sum(node.sampled_returns) / len(node.sampled_returns)
    else:
        raise Exception(f"Unknown tree search mode {mode}")

def combinations(space):
    if isinstance(space, spaces.Discrete):
        return range(space.n)
    elif isinstance(space, spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

def mcts_tree_policy(ag, children):
    return random.choice(children)

def mcts_procedure(ag, tree_policy, env, done, root=None, rollout_weight=1., term_cond=None, ts_mode="best"):
    """
    Compute the entire MCTS procedure wrt to the selected tree policy.
    Funciton tree_policy is a function taking an agent + a list of ChanceNodes as argument
    and returning the one chosen by the tree policy.
    """
    decision_node_num = 0
    if root is not None:
        # if using existing tree, making sure the root is updated correctly
        assert root.state == env.state
    else:
        # create an empty tree
        root = DecisionNode(None, env.state, ag.action_space.copy(), done, dp=ag.dp, id=decision_node_num)
        decision_node_num += 1

    # make sure the rollout number is at least one, and is at most ag.rollouts
    if rollout_weight > 1:
        warnings.warn("How come rollout_weight > 1? Setting to 1.")

    rollouts = np.clip(int(ag.rollouts * rollout_weight), 1, ag.rollouts)

    print("Performing rollouts.")
    for _ in tqdm(range(rollouts)):
        if term_cond is not None and term_cond():
            break
        rewards = [] # Rewards collected along the tree for the current rollout
        node = root # Current node
        terminal = done

        # Selection
        select = True
        while select:
            if (type(node) == DecisionNode): # DecisionNode
                if node.is_terminal:
                    select = False # Selected a terminal DecisionNode
                else:
                    node = tree_policy(ag, node.children) # Shun: move down the tree, node is now a ChanceNode
            else: # ChanceNode
                state_p, reward, terminal = env.transition(node.parent.state, node.action, ag.is_model_dynamic)
                rewards.append(reward)

                new_state = True
                for i in range(len(node.children)):
                    # Note that for deterministic transitions, node.children contains at most one child
                    if env.equality_operator(node.children[i].state, state_p):
                        # Shun: state_p already in the tree, point node to the corresponding Decision Node
                        node = node.children[i]
                        new_state = False
                        break
                if new_state:
                    select = False # Selected a ChanceNode

        # Expansion
        # If node is a decision node, then it must be a terminal node, do nothing here
        if type(node) == ChanceNode:
            node.children.append(DecisionNode(node, state_p, ag.action_space.copy(), terminal, dp=ag.dp, id=decision_node_num))
            decision_node_num += 1
            node = node.children[-1]

        # Evaluation
        # now `rewards` collected all rewards in the ChanceNodes above this node
        assert(type(node) == DecisionNode)
        state = node.state
        if ag.dp is None:
            t = 0
            estimate = 0
            while (not terminal) and (t < ag.horizon):
                action = env.action_space.sample()
                state, reward, terminal = env.transition(state, action, ag.is_model_dynamic)
                estimate += reward * (ag.gamma**t)
                t += 1
        else:
            if not node.is_terminal:
                if ag.dp.use_value:
                    state = ag.dp.get_short_horizon_sequence(state)
                    # this state may not be a complete program, call the pre-trained value model to get an estimate
                    estimate = ag.dp.get_value(state)
                else:
                    # follow the default policy to get a terminal state
                    state = ag.dp.get_predict_sequence(state)
                    estimate = env.get_reward(state)

                    # save this information for demo
                    node.info['complete_program'] = state
            else:
                # the rewards are defined on terminating actions, the terminal states have no rewards
                estimate = 0

        # Backpropagation
        node.visits += 1
        node = node.parent
        assert(type(node) == ChanceNode)
        while node:
            if len(rewards) != 0:
                estimate = rewards.pop() + ag.gamma * estimate
            node.sampled_returns.append(estimate)
            node.parent.visits += 1
            node = node.parent.parent

        # should finish backpropagating all the rewards back
        assert len(rewards) == 0

    return max(root.children, key=lambda n: chance_node_value(n, mode=ts_mode)).action, root

class DecisionNode:
    """
    Decision node class, labelled by a state

    Args:
        dp: default policy, used to prioritize and filter possible actions
    """
    def __init__(self, parent, state, possible_actions=[], is_terminal=False, dp=None, id=None):
        self.id = id
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None: # Root node
            self.depth = 0
        else: # Non root node
            self.depth = parent.depth + 1
        if dp is None:
            self.possible_actions = possible_actions
            random.shuffle(self.possible_actions)

            # if no default policy is provided, assume selection probability is uniform
            self.action_scores = [1.0 / len(self.possible_actions)] * len(self.possible_actions)
        else:
            # get possible actions from dp
            # default policy suggests what children to consider
            top_k_predict, top_k_scores = dp.get_top_k_predict(self.state)

            self.possible_actions = top_k_predict
            self.action_scores = top_k_scores

        # populate its children
        self.children = [ChanceNode(self, (act, score)) for act, score in zip(self.possible_actions, self.action_scores)]

        self.explored_children = 0
        # this decision node should be visited at least once, otherwise p-uct makes no sense for this node
        self.visits = 1
        # used to save any information of the state
        # we use this for saving complete programs generated from it
        self.info = {}

    def is_fully_expanded(self):
        return all([child.expanded() for child in self.children])


class ChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """
    def __init__(self, parent, action_and_score):
        self.parent = parent
        self.action = action_and_score[0]
        self.depth = parent.depth
        self.children = []
        self.prob = action_and_score[1] # the probability that this action should be token, provided by default policy
        self.sampled_returns = []

    def expanded(self):
        return len(self.children) > 0


class MCTS(object):
    """
    MCTS agent
    """
    def __init__(self, action_space, rollouts=100, horizon=100, gamma=0.9, is_model_dynamic=True):
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(combinations(action_space))
        else:
            self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.is_model_dynamic = is_model_dynamic

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p == None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p,[spaces.discrete.Discrete, int, int, float, bool])
            self.__init__(p[0], p[1], p[2], p[3], p[4])


    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying MCTS agent:')
        print('Action space       :', self.action_space)
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('Is model dynamic   :', self.is_model_dynamic)

    def act(self, env, done):
        opt_act, _, = mcts_procedure(self, mcts_tree_policy, env, done)
        return opt_act


def update_root(ag, act, state_p):
    root_updated = False
    for chance_node in ag.root.children:
        if act == chance_node.action:
            for decision_node in chance_node.children:
                if decision_node.state == state_p:
                    ag.root = decision_node
                    root_updated = True
                    break

    if not root_updated:
        raise Exception("root update fails, can't find the next state, action pair in tree.")


def pre_order_traverse(
        decision_node: DecisionNode,
        decision_node_fn=lambda n, d: None,
        chance_node_fn=lambda n, d: None,
        depth=0):
    """
    Postorder traversal of the tree rooted at state
    Apply fn once visited
    """
    decision_node_fn(decision_node, depth)

    for chance_node in decision_node.children:
        chance_node_fn(chance_node, depth)
        for next_decision_node in chance_node.children:
            pre_order_traverse(next_decision_node, decision_node_fn, chance_node_fn, depth + 1)

def get_all_decision_nodes(root: DecisionNode):
    """
    Get all decision nodes in the tree
    """
    decision_nodes = []
    pre_order_traverse(root, decision_node_fn=lambda n, d: decision_nodes.append(n))
    return decision_nodes

def plot_tree(root: DecisionNode, env, filename):
    """
    Plot the tree rooted at root
    """
    tokenizer = env.tokenizer

    def printer(node: ChanceNode, depth):
        # print the average return of the *parent* of this state
        # (this is easier to implement than printing all its children nodes)
        print("\t" * depth, repr(tokenizer.decode(node.action)), 'p', node.prob, 'q', chance_node_value(node), 'len(returns)', len(node.sampled_returns))

    pre_order_traverse(root, chance_node_fn=printer)

    # plot the tree
    G = nx.DiGraph()
    G.add_node(root.id, label='<PD>')

    def add_node(node: ChanceNode, depth):
        if len(node.children) > 0:
            child_id = node.children[0].id
            parent_id = node.parent.id

            G.add_node(child_id)
            G.add_edge(parent_id, child_id, label=repr(tokenizer.decode(node.action)))

    pre_order_traverse(root, chance_node_fn=add_node)

    plt.figure(figsize=(15, 15))

    pos = hierarchy_pos(G, root=root.id)
    nx.draw(G, pos, with_labels=True)

    edge_labels = nx.get_edge_attributes(G, 'label')
    # plot labels on the edges horizontally
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, rotate=False)

    plt.savefig(filename + '.pdf', format="pdf")
    plt.close()

def convert_to_json(root: DecisionNode, env, selected_act):
    """
    Save the information of children of root into a list.
    Does not distinguish layers. So works when the tree only expands one level.
    """
    ret = []

    def get_info(node: ChanceNode, depth):
        if node.action == env.terminal_token:
            # terminal state has no complete_program attribute, since the program is already complete
            complete_program = env.convert_state_to_program(node.children[0].state)
        else:
            complete_program = env.convert_state_to_program(node.children[0].info['complete_program'])

        info = {'token': env.tokenizer.decode(node.action),
                'state': env.convert_state_to_program(node.children[0].state),
                'selected': node.action == selected_act,
                'score': chance_node_value(node),
                'complete_program': complete_program}
        ret.append(info)

    pre_order_traverse(root, chance_node_fn=get_info)
    return ret