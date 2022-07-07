import copy
import gym
import itertools
import json
import numpy as np
from kaggle_environments import make
import math
from pathlib import Path
from queue import Queue, Empty
import random
from subprocess import Popen, PIPE
import sys
from threading import Thread
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from from_isaiah.lux.game import Game
from from_lux.game_objects import Unit, CityTile
from from_isaiah.lux_gym.act_spaces import BaseActSpace, ACTION_MEANINGS
from from_isaiah.lux_gym.obs_spaces import BaseObsSpace
from from_isaiah.lux_gym.reward_spaces import BaseRewardSpace
from from_isaiah.lux_gym.wrappers import RewardSpaceWrapper,VecEnv,LoggingEnv
from utility_constants import MAX_BOARD_SIZE
from kaggle_environments.agent import Agent
# In case dir_path is removed in production environment
try:
    from kaggle_environments.envs.lux_ai_2021.lux_ai_2021 import dir_path as DIR_PATH
except Exception:
    DIR_PATH = None


"""
def _cleanup_dimensions_factory(dimension_process: Popen) -> NoReturn:
    def cleanup_dimensions():
        if dimension_process is not None:
            dimension_process.kill()
    return cleanup_dimensions
"""


def _enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def _generate_pos_to_unit_dict(game_state: Game) -> Dict[Tuple, Optional[Unit]]:
    pos_to_unit_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for unit in reversed(player.units):
            pos_to_unit_dict[(unit.pos.x, unit.pos.y)] = unit

    return pos_to_unit_dict


def _generate_pos_to_city_tile_dict(game_state: Game) -> Dict[Tuple, Optional[CityTile]]:
    pos_to_city_tile_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for city in player.cities.values():
            for city_tile in city.citytiles:
                pos_to_city_tile_dict[(city_tile.pos.x, city_tile.pos.y)] = city_tile

    return pos_to_city_tile_dict


# noinspection PyProtectedMember
class LuxEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            reward_space:BaseRewardSpace,
            configuration: Optional[Dict[str, Any]] = None,
            seed: Optional[int] = None,
            run_game_automatically: bool = True,
            restart_subproc_after_n_resets: int = 100
    ):
        super(LuxEnv, self).__init__()
        self.obs_space = obs_space
        self.action_space = act_space
        self.default_reward_space = reward_space
        self.observation_space = obs_space#self.obs_space.get_obs_spec()
        self.board_dims = configuration['width'],configuration['height']
        self.run_game_automatically = run_game_automatically
        self.restart_subproc_after_n_resets = restart_subproc_after_n_resets

        self.game_state = Game()
        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = make("lux_ai_2021").configuration
            # 2: warnings, 1: errors, 0: none
            self.configuration["loglevel"] = 0
        if seed is not None:
            self.seed(seed)
        elif "seed" not in self.configuration:
            self.seed()


        self.done = False
        self.info = {}
        self.pos_to_unit_dict = dict()
        self.pos_to_city_tile_dict = dict()
        self.reset_count = 0

        self._dimension_process = None
        self._q = None
        self._t = None
        self._restart_dimension_process()

    def _restart_dimension_process(self) -> NoReturn:
        if self._dimension_process is not None:
            self._dimension_process.kill()
        if self.run_game_automatically:
            # 1.1: Initialize dimensions in the background
            self._dimension_process = Popen(
                ["node", str(Path(DIR_PATH) / "dimensions/main.js")],
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE
            )
            self._q = Queue()
            self._t = Thread(target=_enqueue_output, args=(self._dimension_process.stdout, self._q))
            self._t.daemon = True
            self._t.start()
            # atexit.register(_cleanup_dimensions_factory(self._dimension_process))

    def reset(self, observation_updates: Optional[List[str]] = None) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        self.game_state = Game()
        self.reset_count = (self.reset_count + 1) % self.restart_subproc_after_n_resets
        # There seems to be a gradual memory leak somewhere, so we restart the dimension process every once in a while
        if self.reset_count == 0:
            self._restart_dimension_process()
        if self.run_game_automatically:
            assert observation_updates is None, "Game is being run automatically"
            # 1.2: Initialize a blank state game if new episode is starting
            self.configuration["seed"] += 1
            initiate = {
                "type": "start",
                "agent_names": [],  # unsure if this is provided?
                "config": self.configuration
            }
            self._dimension_process.stdin.write((json.dumps(initiate) + "\n").encode())
            self._dimension_process.stdin.flush()
            agent1res = json.loads(self._dimension_process.stderr.readline())
            # Skip agent2res and match_obs_meta
            _ = self._dimension_process.stderr.readline(), self._dimension_process.stderr.readline()

            self.game_state._initialize(agent1res)
            self.game_state._update(agent1res[2:])
        else:
            assert observation_updates is not None, "Game is not being run automatically"
            self.game_state._initialize(observation_updates)
            self.game_state._update(observation_updates[2:])

        self.done = False
        self.board_dims = (self.game_state.map_width, self.game_state.map_height)
        self.observation_space = self.obs_space.get_obs_spec(self.board_dims)
        self.info = {
            "actions_taken": {
                key: np.zeros(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
                for key, space in self.action_space.get_action_space(self.board_dims).spaces.items()
            },
            "available_actions_mask": {
                key: np.ones(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
                for key, space in self.action_space.get_action_space(self.board_dims).spaces.items()
            }
        }
        self._update_internal_state()

        return self.get_obs_reward_done_info()

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        if self.run_game_automatically:
            actions_processed, actions_taken = action,1#self.process_actions(action) change
            self._step(actions_processed)
            self.info["actions_taken"] = actions_taken
        self._update_internal_state()

        return self.get_obs_reward_done_info()

    def manual_step(self, observation_updates: List[str]) -> NoReturn:
        assert not self.run_game_automatically
        self.game_state._update(observation_updates)

    def get_obs_reward_done_info(self) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        rewards = self.default_reward_space.compute_rewards(game_state=self.game_state, done=self.done)
        return self.observation_space.observation(self.game_state), rewards, self.done, copy.copy(self.info)

    def process_actions(self, action: Dict[str, np.ndarray]) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        return self.action_space.process_actions(
            action,
            self.game_state,
            self.board_dims,
            self.pos_to_unit_dict
        )

    def _step(self, action: List[List[str]]) -> NoReturn:
        # 2.: Pass in actions (json representation along with id of who made that action),
        #       and agent information (id, status) to dimensions via stdin
        assert len(action) == 2
        # TODO: Does dimension process state need to include info other than actions?
        state = [{'action': a} for a in action]
        self._dimension_process.stdin.write((json.dumps(state) + "\n").encode())
        self._dimension_process.stdin.flush()

        # 3.1 : Receive and parse the observations returned by dimensions via stdout
        agent1res = json.loads(self._dimension_process.stderr.readline())
        # Skip agent2res and match_obs_meta
        _ = self._dimension_process.stderr.readline(), self._dimension_process.stderr.readline()
        self.game_state._update(agent1res)

        # Check if done
        match_status = json.loads(self._dimension_process.stderr.readline())
        self.done = match_status["status"] == "finished"

        while True:
            try:
                line = self._q.get_nowait()
            except Empty:
                # no standard error received, break
                break
            else:
                # standard error output received, print it out
                print(line.decode(), file=sys.stderr, end='')

    def _update_internal_state(self) -> NoReturn:
        self.pos_to_unit_dict = _generate_pos_to_unit_dict(self.game_state)
        self.pos_to_city_tile_dict = _generate_pos_to_city_tile_dict(self.game_state)
        self.info["available_actions_mask"] = self.action_space.get_available_actions_mask(
            self.game_state,
            self.board_dims,
            self.pos_to_unit_dict,
            self.pos_to_city_tile_dict
        )

    def seed(self, seed: Optional[int] = None) -> NoReturn:
        if seed is not None:
            # Seed is incremented on reset()
            self.configuration["seed"] = seed - 1
        else:
            self.configuration["seed"] = math.floor(random.random() * 1e9)

    def get_seed(self) -> int:
        return self.configuration["seed"]

    def render(self, mode='human'):
        raise NotImplementedError('LuxEnv rendering is not implemented. Use the Lux visualizer instead.')
class LuxEnv2(LuxEnv):
    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            reward_space: BaseRewardSpace,
            configuration: Optional[Dict[str, Any]] = None,
            seed: Optional[int] = None,
            run_game_automatically: bool = True,
            restart_subproc_after_n_resets: int = 100
    ):
        super(LuxEnv2, self).__init__(act_space,
            obs_space,
            reward_space,
            configuration)

        self.sub_environment=make("lux_ai_2021",configuration=configuration)

        self.sub_environment.debug=True
        self.game_state=Game()
        self.game_state._initialize([0, str(self.configuration['width'] )+" " + str( self.configuration['height'])])

    def add_agents(self,agents:list):
        self.sub_environment.agents = {'simp': agents[0], 'simp2':agents[1] }
        #self.runner = self.__agent_runner([agents[0], agents[1]])
        #pass



    def reset(self):
        #self.sub_environment.reset()

        temp=self.sub_environment.reset()
        self.info['remainingOverageTime']=temp[0]['observation']['remainingOverageTime'],temp[0]['observation']['remainingOverageTime']
        self.game_state._update(temp[0]['observation']['updates'])
        return self.get_obs_reward_done_info()

        #self.game_state._update([0, str(self.configuration['width']) + " " + str(self.configuration['height'])])



        return actions, logs
    def step(self,actions):

        temp=self.sub_environment.step(actions, self.sub_environment.logs)
        self.game_state._update(temp[0]['observation']['updates'])
        if self.sub_environment.done:self.done=True

        return self.get_obs_reward_done_info()

        c=0
def __agent_runner(env, agents):
        agents = [
        Agent(agent,env)
        if agent is not None
        else None
        for agent in agents
    ]
        return agents
def get_action(runner,observation):
        class observation():
            remainingOverageTime=100000000

        action1, log1 = runner[0].act(observation())
        action2, log2 = runner[1].act(observation())
        actions=[action1,action2]

        logs=[log1,log2]
        return actions, logs

if __name__ =='__main__':
    import torch
    #from from_isaiah.lux_env import LuxEnv2
    from from_isaiah.lux_gym.act_spaces import BasicActionSpace
    from from_isaiah.lux_gym import obs_spaces, act_spaces, reward_spaces,wrappers
    from torchbeast.monobeast import train
    #from from_isaiah.rl_agent import rl_agent
    from types import SimpleNamespace
    import yaml
    from vivek_agent import agent1

    with open('/home/pooja/PycharmProjects/lux_ai/codes/config.yaml', 'r') as f:
        config = SimpleNamespace(**yaml.safe_load(f))


    act_space = act_spaces.__dict__[config.act_space]()  # .get_action_space()
    obs_space = obs_spaces.__dict__[config.obs_space]()  # .get_obs_spec((12,12))
    reward_space = reward_spaces.__dict__[config.reward_space]()  # .get_obs_spec((12,12))
    configuration = {"seed": 59353, "loglevel": 2, "annotations": True, "width": 12, "height": 12}
    # configuration=make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True,"width":12, "height":12}, debug=True)
    env = LuxEnv2(obs_space=obs_space, act_space=act_space, reward_space=reward_space, configuration=configuration)
    #env.add_agents([lambda y:['m u2 n'],lambda y:['m u2 n']])
    agents=[lambda y:['m u_1 n'],lambda y:['m u_1 n']]
    runner=__agent_runner(env.sub_environment, agents)

    env.observation_space = env.obs_space.wrap_env(env.observation_space)
    env = wrappers.LoggingEnv(wrappers.RewardSpaceWrapper(env, reward_space))





    obs=env.reset()

    #action, log = get_action(runner,obs[0])

    while not env.sub_environment.done:

        #temp= env.observation_space.observation(obs[0])
        #temp=env.sub_environment.agents['simp'].see(obs[0])
        actions,logs=get_action(runner,obs[0])
        env.sub_environment.logs=logs
        obs=env.step(actions)
    x=env.sub_environment.render(mode='html')
    f = open('/home/pooja/PycharmProjects/lux_ai/outputs/GFG.html', 'w')
    f.write(x)
    f.close
    p=0
