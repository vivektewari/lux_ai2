import copy
import gym
import numpy as np
import torch
from typing import Dict, List, NoReturn, Optional, Tuple, Union
from visdom import Visdom
from .act_spaces import ACTION_MEANINGS
from .lux_env import LuxEnv
from .reward_spaces import BaseRewardSpace
from ..utility_constants import MAX_BOARD_SIZE



class PadFixedShapeEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, max_board_size: Tuple[int, int] = MAX_BOARD_SIZE):
        super(PadFixedShapeEnv, self).__init__(env)
        self.max_board_size = max_board_size
        self.observation_space = self.unwrapped.obs_space.get_obs_spec(max_board_size)
        self.input_mask = np.zeros((1,) + max_board_size, dtype=bool)
        self.input_mask[:, :self.orig_board_dims[0], :self.orig_board_dims[1]] = True

    def _pad(self, x: Union[Dict, np.ndarray]) -> Union[dict, np.ndarray]:
        if isinstance(x, dict):
            return {key: self._pad(val) for key, val in x.items()}
        elif x.ndim == 4 and x.shape[2:4] == self.orig_board_dims:
            return np.pad(x, pad_width=self.base_pad_width, constant_values=0.)
        elif x.ndim == 5 and x.shape[2:4] == self.orig_board_dims:
            return np.pad(x, pad_width=self.base_pad_width + ((0, 0),), constant_values=0.)
        else:
            return x

    def observation(self, observation: Dict[str, Union[Dict, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            key: self._pad(val) for key, val in observation.items()
        }

    def info(self, info: Dict[str, Union[Dict, np.ndarray]]) -> Dict[str, np.ndarray]:
        info = {
            key: self._pad(val) for key, val in info.items()
        }
        assert "input_mask" not in info.keys()
        info["input_mask"] = self.input_mask
        return info

    def reset(self, **kwargs):
        obs, reward, done, info = super(PadFixedShapeEnv, self).reset(**kwargs)
        self.input_mask[:] = 0
        self.input_mask[:, :self.orig_board_dims[0], :self.orig_board_dims[1]] = 1
        return self.observation(obs), reward, done, self.info(info)

    def step(self, action: Dict[str, np.ndarray]):
        action = {
            key: val[..., :self.orig_board_dims[0], :self.orig_board_dims[1]] for key, val in action.items()
        }
        obs, reward, done, info = super(PadFixedShapeEnv, self).step(action)
        return self.observation(obs), reward, done, self.info(info)

    @property
    def orig_board_dims(self) -> Tuple[int, int]:
        return self.unwrapped.board_dims

    @property
    def base_pad_width(self):
        return (
            (0, 0),
            (0, 0),
            (0, self.max_board_size[0] - self.orig_board_dims[0]),
            (0, self.max_board_size[1] - self.orig_board_dims[1])
        )


class RewardSpaceWrapper(gym.Wrapper):
    def __init__(self, env: LuxEnv, reward_space: BaseRewardSpace):
        super(RewardSpaceWrapper, self).__init__(env)
        self.reward_space = reward_space

    def _get_rewards_and_done(self) -> Tuple[Tuple[float, float], bool]:
        rewards, done = self.reward_space.compute_rewards_and_done(self.unwrapped.game_state, self.done)
        if self.unwrapped.done and not done:
            raise RuntimeError("Reward space did not return done, but the lux engine is done.")
        self.unwrapped.done = done
        return rewards, done
    def get_obs_reward_done_info(self,**kwargs) -> Tuple[Tuple[float, float], bool]:
        obs, reward, done, info = super(RewardSpaceWrapper, self).get_obs_reward_done_info(**kwargs)
        rewards, done = self.reward_space.compute_rewards_and_done(self.unwrapped.game_state, self.done)
        if self.unwrapped.done and not done:
            raise RuntimeError("Reward space did not return done, but the lux engine is done.")
        self.unwrapped.done = done
        return (obs,rewards, done,info)

    def reset(self, **kwargs):
        obs, _, _, info = super(RewardSpaceWrapper, self).reset(**kwargs)
        return (obs, *self._get_rewards_and_done(), info)

    def step(self, action):
        obs, _, _, info = super(RewardSpaceWrapper, self).step(action)
        return (obs, *self._get_rewards_and_done(), info)



class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='default'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
class LoggingEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(LoggingEnv, self).__init__(env)
        self.logs=None
        #log_add: for addition in steps step 1/3
        self.track_dict={'delta':[],'value':[],'action_prob':[],'city_action_research':[],'unit_build':[],'bcity':[],'city_no_action':[],'unit_no_action':[],'move_action':[], 'unit_no_action_prob':[], 'city_no_action_prob':[],'blank_space_no_act_prob': [],'max_model_output':[],'rewards':[],'bs_loss':[],'entropy':[]}

        # self.delta=[]
        # self.value = []
        # self.action_prob = []

        self.max_size=1
        self.vals_peak = {}
        self.reward_sums = [0., 0.]
        self.actions_distributions = {
            f"{space}.{meaning}": 0.
            for space, action_meanings in ACTION_MEANINGS.items()
            for meaning in action_meanings
        }
        # TODO: Resource mining % like in visualizer?
        # self.resource_count = {"wood", etc...}
        # TODO: Fuel metric?

    def add_logs_and_visulizer(self,logs=None,visualizer=None):

        if visualizer is None:self.visualizer=VisdomLinePlotter()
        else:self.visualizer=visualizer
        if logs is None:
            #self.visualizer.viz.
            #log_add: step2/3
            self.logs = {
                "step": [],
                "city_tiles": [],
                "separate_cities": [],
                "workers": [],
                "carts": [],
                "research_points": [],
                "rewards": [],
                "game_win":[],
                "value":[],
                "delta":[],
                "action_prob":[],
                'city_action_research': [],
                'unit_build': [],
                'bcity': [],
                'city_no_action': [],
                'unit_no_action': [],
                'move_action':[],
                'unit_no_action_prob':[],
                'city_no_action_prob':[],
                'blank_space_no_act_prob': [],
                'max_model_output':[],
                'bs_loss':[],
                'entropy':[]

            }
        else :self.logs=logs

    def get_obs_reward_done_info(self,**kwargs) :
        obs, reward, done, info = self.unwrapped.get_obs_reward_done_info(**kwargs)
        return obs, reward, done, self.info(info,reward)


    def info(self, info: Dict[str, np.ndarray], rewards: List[int]) -> Dict[str, np.ndarray]:
        if not self.done: return {}
        info = copy.copy(info)
        self.rewards_copy=rewards
        return info
    def prepare_logs(self):
        rewards=self.rewards_copy
        game_state = self.env.unwrapped.game_state

        if game_state.players[0].city_tile_count>game_state.players[1].city_tile_count:win=[1,0]
        elif game_state.players[0].city_tile_count < game_state.players[1].city_tile_count: win = [0, 1]
        elif len(game_state.players[0].units) > len(game_state.players[1].units):win=[1,0]
        elif len(game_state.players[0].units) < len(game_state.players[1].units):win = [0, 1]
        else :win=[0,0]

        #log_add: step3/3
        logs = {
            "step": [game_state.turn],
            "city_tiles": [p.city_tile_count for p in game_state.players],
            "separate_cities": [len(p.cities) for p in game_state.players],
            "workers": [sum(u.is_worker() for u in p.units) for p in game_state.players],
            "carts": [sum(u.is_cart() for u in p.units) for p in game_state.players],
            "research_points": [p.research_points for p in game_state.players],
            "rewards": [np.sum(self.track_dict['rewards'])],
            "game_win": win,
            "value":[self.track_dict['value'][0],np.max(np.absolute(self.track_dict['value']))],
            "delta":[np.mean(self.track_dict['delta']),np.max(np.absolute(self.track_dict['delta']))],
            "action_prob": [np.mean(self.track_dict['action_prob']), np.max(self.track_dict['action_prob'])],
            'city_action_research': [np.mean(self.track_dict['city_action_research'])],
            'unit_build': [np.mean(self.track_dict['unit_build'])],
               'bcity':[np.mean(self.track_dict['bcity'])],
            'city_no_action': [np.mean(self.track_dict['city_no_action'])],
            'unit_no_action': [np.mean(self.track_dict['unit_no_action'])],
            'move_action':[np.mean(self.track_dict['move_action'])],
            'unit_no_action_prob':[np.mean(self.track_dict['unit_no_action_prob'])],
            'city_no_action_prob': [np.mean(self.track_dict['city_no_action_prob'])],
            'blank_space_no_act_prob': [np.mean(self.track_dict['blank_space_no_act_prob'])],
            'max_model_output':[np.mean(self.track_dict['max_model_output'])],
            'bs_loss':self.track_dict['bs_loss'],
            'entropy':self.track_dict['entropy'],

        }

        self.vals_peak = {
            key: np.maximum(val, logs[key]) for key, val in self.vals_peak.items()
        }
        for key, val in self.vals_peak.items():
            logs[f"{key}_peak"] = val.copy()
            logs[f"{key}_final"] = logs[key]
            del logs[key]

        self.reward_sums = [r + s for r, s in zip(rewards, self.reward_sums)]
        # logs["mean_cumulative_rewards"] = [np.mean(self.reward_sums)]
        # logs["mean_cumulative_reward_magnitudes"] = [np.mean(np.abs(self.reward_sums))]
        # logs["max_cumulative_rewards"] = [np.max(self.reward_sums)]

        for key in self.logs.keys():
            if len(self.logs[key])>=self.max_size:
                self.logs[key].pop(0)
            self.logs[key].append(logs[key])
        #return self.logs,self.visualizer

        #ToDo
        # actions_taken = self.env.unwrapped.action_space.actions_taken_to_distributions(info["actions_taken"])
        # self.actions_distributions = {
        #     f"{space}.{act}": self.actions_distributions[f"{space}.{act}"] + n
        #     for space, dist in actions_taken.items()
        #     for act, n in dist.items()
        # }
        # logs.update({f"ACTIONS_{key}": val for key, val in self.actions_distributions.items()})
        #
        # info.update({f"LOGGING_{key}": np.array(val, dtype=np.float32) for key, val in logs.items()})
        # # Add any additional info from the reward space
        # info.update(self.reward_space.get_info())





    def reset(self, **kwargs):
        obs, reward, done, info = super(LoggingEnv, self).reset(**kwargs)
        self._reset_peak_vals()
        self.reward_sums = [0., 0.]
        self.actions_distributions = {
            key: 0. for key in self.actions_distributions.keys()
        }
        return obs, reward, done, self.info(info, reward)

    def step(self, action: Dict[str, np.ndarray]):
        obs, reward, done, info = super(LoggingEnv, self).step(action)
        return obs, reward, done, self.info(info, reward)

    def _reset_peak_vals(self) -> NoReturn:
        self.vals_peak = {
            key: np.array([0., 0.])
            for key in [
                "city_tiles",
                "separate_cities",
                "workers",
                "carts",
            ]
        }


class VecEnv(gym.Env):
    def __init__(self, envs: List[gym.Env]):
        self.envs = envs
        self.last_outs = [() for _ in range(len(self.envs))]
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.envs[0], name)

    @staticmethod
    def _stack_dict(x: List[Union[Dict, np.ndarray]]) -> Union[Dict, np.ndarray]:
        if isinstance(x[0], dict):
            return {key: VecEnv._stack_dict([i[key] for i in x]) for key in x[0].keys()}
        else:
            return np.stack([arr for arr in x], axis=0)

    @staticmethod
    def _vectorize_env_outs(env_outs: List[Tuple]) -> Tuple:
        obs_list, reward_list, done_list, info_list = zip(*env_outs)
        obs_stacked = VecEnv._stack_dict(obs_list)
        reward_stacked = np.array(reward_list)
        done_stacked = np.array(done_list)
        info_stacked = VecEnv._stack_dict(info_list)
        return obs_stacked, reward_stacked, done_stacked, info_stacked

    def reset(self, force: bool = False, **kwargs):
        if force:
            # noinspection PyArgumentList
            self.last_outs = [env.reset(**kwargs) for env in self.envs]
            return VecEnv._vectorize_env_outs(self.last_outs)

        for i, env in enumerate(self.envs):
            # Check if env finished
            if self.last_outs[i][2]:
                # noinspection PyArgumentList
                self.last_outs[i] = env.reset(**kwargs)
        return VecEnv._vectorize_env_outs(self.last_outs)

    def step(self, action: Dict[str, np.ndarray]):
        actions = [
            {key: val[i] for key, val in action.items()} for i in range(len(self.envs))
        ]
        self.last_outs = [env.step(a) for env, a in zip(self.envs, actions)]
        return VecEnv._vectorize_env_outs(self.last_outs)

    def render(self, idx: int, mode: str = "human", **kwargs):
        # noinspection PyArgumentList
        return self.envs[idx].render(mode, **kwargs)

    def close(self):
        return [env.close() for env in self.envs]

    def seed(self, seed: Optional[int] = None) -> list:
        if seed is not None:
            return [env.seed(seed + i) for i, env in enumerate(self.envs)]
        else:
            return [env.seed(seed) for i, env in enumerate(self.envs)]

    @property
    def unwrapped(self) -> List[gym.Env]:
        return [env.unwrapped for env in self.envs]

    @property
    def action_space(self) -> List[gym.spaces.Dict]:
        return [env.action_space for env in self.envs]

    @property
    def observation_space(self) -> List[gym.spaces.Dict]:
        return [env.observation_space for env in self.envs]

    @property
    def metadata(self) -> List[Dict]:
        return [env.metadata for env in self.envs]


class PytorchEnv(gym.Wrapper):
    def __init__(self, env: Union[gym.Env, VecEnv], device: torch.device = torch.device("cpu")):
        super(PytorchEnv, self).__init__(env)
        self.device = device
        
    def reset(self, **kwargs):
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).reset(**kwargs)])
        
    def step(self, action: Dict[str, torch.Tensor]):
        action = {
            key: val.cpu().numpy() for key, val in action.items()
        }
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).step(action)])

    def _to_tensor(self, x: Union[Dict, np.ndarray]) -> Dict[str, Union[Dict, torch.Tensor]]:
        if isinstance(x, dict):
            return {key: self._to_tensor(val) for key, val in x.items()}
        else:
            return torch.from_numpy(x).to(self.device, non_blocking=True)


class DictEnv(gym.Wrapper):
    @staticmethod
    def _dict_env_out(env_out: tuple) -> dict:
        obs, reward, done, info = env_out
        assert "obs" not in info.keys()
        assert "reward" not in info.keys()
        assert "done" not in info.keys()
        return dict(
            obs=obs,
            reward=reward,
            done=done,
            info=info
        )

    def reset(self, **kwargs):
        return DictEnv._dict_env_out(super(DictEnv, self).reset(**kwargs))

    def step(self, action):
        return DictEnv._dict_env_out(super(DictEnv, self).step(action))
