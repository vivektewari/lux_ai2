import copy

import torch
from kaggle_environments import make
from monobeast_lux_ai import create_env
from models.losses import with_1_step_grad
import numpy as np
import models.conv_blocks as conv_blocks
from from_isaiah.lux_env import LuxEnv2
from from_isaiah.lux_gym.act_spaces import BasicActionSpace
from from_isaiah.lux_gym import obs_spaces, act_spaces, reward_spaces, wrappers
from from_lux.from_kaggle_env import __agent_runner
from kaggle_environments.agent import Agent
from from_vivek.brain import Brain_v1,Brain_v2
from from_vivek.env import LuxEnv

from types import SimpleNamespace
import yaml
from vivek_agent import agent_v2
from utils.visualizer import Visualizer
from collections import deque
from auxilary import visdom_print,VisdomLinePlotter
domains = [0 for i in range(10)]
my_queue = deque(domains)


    ###################################################

with open('./config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))
flags = config
model_ = conv_blocks.__dict__[config.nn_model_name]


#conv_blocks.count_parameters(model1)
brain1=Brain_v2( discounting=0.9,lambda_w=0.4,lambda_t=0.4,alpha_w=0.002,alpha_t=0.0012,model=None)
brain2=Brain_v2( discounting=0.9,lambda_w=0.4,lambda_t=0.4,alpha_w=0.002,alpha_t=0.0012,model=None)
temp=config.savedir
config.savedir=None
if True:

    brain1.load_model(model_constructor=model_,model_params=config.model_params,dir=config.savedir,identifier=1,eps=150)
    brain2.load_model(model_constructor=model_, model_params=config.model_params,
                      dir=config.savedir,identifier=2,eps=150)
config.savedir=temp
# configuration=make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True,"width":12, "height":12}, debug=True)
win_count_0=0
game_played=0


check_step=-2
visualizer = Visualizer()
model_player=0


act_space = act_spaces.__dict__[config.act_space] # .get_action_space()
obs_space = obs_spaces.__dict__[config.obs_space] # .get_obs_spec((12,12))
reward_space = reward_spaces.__dict__[config.reward_space]()  # .get_obs_spec((12,12))

configuration = {"seed": 1234567, "loglevel": 1, "annotations": True, "width": 12, "height": 12}
env = LuxEnv(obs_space=obs_space(), act_space=act_space(), reward_space=reward_space, configuration=configuration)

env.observation_space = env.obs_space.wrap_env(env=env.observation_space)
env = wrappers.LoggingEnv(wrappers.RewardSpaceWrapper(env, reward_space))


agent1 = agent_v2(player=0, action_space=act_space(default_board_dims=(12, 12)),environment=copy.deepcopy(env),brain=brain1,use_learning=True)

agent2 = agent_v2(player=1, action_space=act_space(default_board_dims=(12, 12)),environment=copy.deepcopy(env),brain=brain2,use_learning=True)

agents = [agent1, agent2]
episodes = 1000
size = 1

logged,visulizer=[None,None],VisdomLinePlotter()
for eps in range(episodes):
    configuration = {"seed": np.random.randint(1,1000000), "loglevel": 1, "annotations": True, "width": 12, "height": 12}
    for ag in agents:
        ag.brain.reset()
        ag.reset_environment(copy.deepcopy(env))
        ag.environment.add_logs_and_visulizer(logged[ag.id], visulizer)
        ag.environment.env.reward_space.add_id(ag.id)



    print("=== Episode {} ===".format(eps))
    kaggle_env = make("lux_ai_2021",
               configuration=configuration,
               debug=True)
    steps = kaggle_env.run(agents)
    logged=[]
    for ag in agents:
        ag.post_game(steps[-1][0]['observation'])
        logged.append(ag.environment.logs)

    visulizer=visdom_print(visulizer, eps,ag.environment.track_dict.keys(),logged[0],logged[1])
    if (eps)%50==0:
        brain1.save_model(config.savedir,eps,1)
        brain2.save_model(config.savedir, eps,2)

        page = kaggle_env.render(mode="html")
        with open(config.renderdir + str(eps)+'game.html', 'w') as f:
            f.write(page)



    print("++++++++++++completed episode {}+++++++++++++++".format(eps))



