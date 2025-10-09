from typing import Any, Callable, Dict, Generator, Optional, Set, Sequence, Type, Tuple, \
    Union
import time
import pickle
import os
import shutil

import jax.numpy as jnp
import jax.random as random
import numpy as np

from pyRDDLGym import RDDLEnv
from pyRDDLGym.core.policy import BaseAgent
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner

Activation = Callable[[jnp.ndarray], jnp.ndarray]
Bounds = Dict[str, Tuple[np.ndarray, np.ndarray]]
Kwargs = Dict[str, Any]
Pytree = Any

class JaxPolicy(BaseAgent):
    '''A container class for a Jax policy trained offline.'''

    use_tensor_obs = True

    def __init__(self, planner: JaxBackpropPlanner,
                 key: Optional[random.PRNGKey] = None,
                 eval_hyperparams: Optional[Dict[str, Any]] = None,
                 params: Optional[Union[str, Pytree]] = None,
                 train_on_reset: bool = False,
                 save_path: Optional[str] = None,
                 **train_kwargs) -> None:
        '''Creates a new JAX offline control policy that is trained once, then
        deployed later.

        :param planner: underlying planning algorithm for optimizing actions
        :param key: the RNG key to seed randomness (derives from clock if not
        provided)
        :param eval_hyperparams: policy hyperparameters to apply for evaluation
        or whenever sample_action is called
        :param params: use the specified policy parameters instead of calling
        planner.optimize(); can be a string pointing to a valid file path where params
        have been saved, or a pytree of parameters
        :param train_on_reset: retrain policy parameters on every episode reset
        :param save_path: optional path to save parameters to
        :param **train_kwargs: any keyword arguments to be passed to the planner
        for optimization
        '''
        self.planner = planner
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))
        self.key = key
        self.eval_hyperparams = eval_hyperparams
        self.train_on_reset = train_on_reset
        self.train_kwargs = train_kwargs
        self.params_given = params is not None
        self.hyperparams_given = eval_hyperparams is not None

        # load the policy from file
        if not self.train_on_reset and params is not None and isinstance(params, str):
            with open(params, 'rb') as file:
                params = pickle.load(file)

        # train the policy
        self.step = 0
        self.callback = None
        if not self.train_on_reset and not self.params_given:
            callback = self.planner.optimize(key=self.key, **self.train_kwargs)
            self.callback = callback
            params = callback['best_params']
            if not self.hyperparams_given:
                self.eval_hyperparams = callback['policy_hyperparams']

            # save the policy
            if save_path is not None:
                with open(save_path, 'wb') as file:
                    pickle.dump(params, file)

        self.params = params

    def sample_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.key, subkey = random.split(self.key)
        actions = self.planner.get_action(
            subkey, self.params, self.step, state, self.eval_hyperparams)
        self.step += 1
        return actions


    def reset(self) -> None:
        self.step = 0

        # train the policy if required to reset at the start of every episode
        if self.train_on_reset and not self.params_given:
            callback = self.planner.optimize(key=self.key, **self.train_kwargs)
            self.callback = callback
            self.params = callback['best_params']
            if not self.hyperparams_given:
                self.eval_hyperparams = callback['policy_hyperparams']

    def evaluate(self, env: RDDLEnv, episodes: int = 1,
                 verbose: bool = False, render: bool = False,
                 seed: Optional[int] = None) -> Dict[str, float]:
        '''Evaluates the current agent on the specified environment by simulating
        roll-outs. Returns a dictionary of summary statistics of the returns
        accumulated on the roll-outs.

        :param env: the environment
        :param episodes: how many episodes (trials) to perform
        :param verbose: whether to print the transition information to console
        at each step of the simulation
        :param render: visualize the domain using the env internal visualizer
        :param seed: optional RNG seed for the environment
        '''

        # check compatibility with environment
        if env.vectorized != self.use_tensor_obs:
            raise ValueError(f'RDDLEnv vectorized flag must match use_tensor_obs '
                             f'of current policy, got {env.vectorized} and '
                             f'{self.use_tensor_obs}, respectively.')

        gamma = env.discount

        # get terminal width
        if verbose:
            width = shutil.get_terminal_size().columns
            sep_bar = '-' * width

        # start simulation
        history = np.zeros((episodes,))
        for episode in range(episodes):

            # restart episode
            total_reward, cuml_gamma = 0.0, 1.0
            self.reset()
            state, _ = env.reset(seed=seed)

            # printing
            if verbose:
                print(f'initial state = \n{self._format(state, width)}')

            # simulate to end of horizon
            for step in range(env.horizon):
                if render:
                    env.render()

                # take a step in the environment
                action = self.sample_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward * cuml_gamma
                cuml_gamma *= gamma
                done = terminated or truncated

                # printing
                if verbose:
                    print(f'{sep_bar}\n'
                          f'step   = {step}\n'
                          f'action = \n{self._format(action, width)}\n'
                          f'state  = \n{self._format(next_state, width)}\n'
                          f'reward = {reward}\n'
                          f'done   = {done}')
                state = next_state
                if done:
                    break

            if verbose:
                print(f'\n'
                      f'episode {episode + 1} ended with return {total_reward}\n'
                      f'{"=" * width}')
            history[episode] = total_reward

            # set the seed on the first episode only
            seed = None

        # summary statistics
        return {
            'mean': np.mean(history),
            'median': np.median(history),
            'min': np.min(history),
            'max': np.max(history),
            'std': np.std(history)
        }


class DeterministicJaxPolicy(JaxPolicy):
    def __init__(self, planner: JaxBackpropPlanner,
                 key: Optional[random.PRNGKey] = None,
                 eval_hyperparams: Optional[Dict[str, Any]] = None,
                 params: Optional[Union[str, Pytree]] = None,
                 train_on_reset: bool = False,
                 save_path: Optional[str] = None,
                 **train_kwargs) -> None:
        # super(DeterministicJaxPolicy, self).__init__(planner, key, eval_hyperparams, params, train_on_reset, save_path, train_kwargs)
        print(train_kwargs)
        super().__init__(planner, key, eval_hyperparams, params, train_on_reset, save_path,
                                                     train_kwargs)

    def sample_action_eval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        actions = self.sample_action(state)
        return actions

    def evaluate(self, env: RDDLEnv, episodes: int = 1,
                 verbose: bool = False, render: bool = False,
                 seed: Optional[int] = None) -> Dict[str, float]:
        '''Evaluates the current agent on the specified environment by simulating
        roll-outs. Returns a dictionary of summary statistics of the returns
        accumulated on the roll-outs.

        :param env: the environment
        :param episodes: how many episodes (trials) to perform
        :param verbose: whether to print the transition information to console
        at each step of the simulation
        :param render: visualize the domain using the env internal visualizer
        :param seed: optional RNG seed for the environment
        '''

        # check compatibility with environment
        if env.vectorized != self.use_tensor_obs:
            raise ValueError(f'RDDLEnv vectorized flag must match use_tensor_obs '
                             f'of current policy, got {env.vectorized} and '
                             f'{self.use_tensor_obs}, respectively.')

        gamma = env.discount

        # get terminal width
        if verbose:
            width = shutil.get_terminal_size().columns
            sep_bar = '-' * width

        # start simulation
        history = np.zeros((episodes,))
        for episode in range(episodes):

            # restart episode
            total_reward, cuml_gamma = 0.0, 1.0
            self.reset()
            state, _ = env.reset(seed=seed)

            # printing
            if verbose:
                print(f'initial state = \n{self._format(state, width)}')

            # simulate to end of horizon
            for step in range(env.horizon):
                if render:
                    env.render()

                # take a step in the environment
                action = self.sample_action_eval(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward * cuml_gamma
                cuml_gamma *= gamma
                done = terminated or truncated

                # printing
                if verbose:
                    print(f'{sep_bar}\n'
                          f'step   = {step}\n'
                          f'action = \n{self._format(action, width)}\n'
                          f'state  = \n{self._format(next_state, width)}\n'
                          f'reward = {reward}\n'
                          f'done   = {done}')
                state = next_state
                if done:
                    break

            if verbose:
                print(f'\n'
                      f'episode {episode + 1} ended with return {total_reward}\n'
                      f'{"=" * width}')
            history[episode] = total_reward

            # set the seed on the first episode only
            seed = None

        # summary statistics
        return {
            'mean': np.mean(history),
            'median': np.median(history),
            'min': np.min(history),
            'max': np.max(history),
            'std': np.std(history)
        }


class StochasticJaxPolicy(DeterministicJaxPolicy):

    def __init__(self, planner: JaxBackpropPlanner,
                 key: Optional[random.PRNGKey] = None,
                 eval_hyperparams: Optional[Dict[str, Any]] = None,
                 params: Optional[Union[str, Pytree]] = None,
                 train_on_reset: bool = False,
                 save_path: Optional[str] = None,
                 **train_kwargs) -> None:
        super(StochasticJaxPolicy, self).__init__(planner, key, eval_hyperparams, params, train_on_reset, save_path,
                                                     train_kwargs)

    def sample_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.key, subkey = random.split(self.key)
        actions = self.planner.get_action(
            subkey, self.params, self.step, state, self.eval_hyperparams)
        self.step += 1
        for key in actions:
            actions[key] = actions[key] + np.random.normal(loc=0, scale=1 , size=actions[key].shape)
        return actions

    def sample_action_eval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.key, subkey = random.split(self.key)
        actions = self.planner.get_action(
            subkey, self.params, self.step, state, self.eval_hyperparams)
        self.step += 1
        return actions

