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

domains = [0 for i in range(10)]
my_queue = deque(domains)


    ###################################################

with open('./config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))
flags = config
model_ = conv_blocks.__dict__[config.nn_model_name]
model1 = model_(**config.model_params)
model_ = conv_blocks.__dict__[config.nn_model_name]

conv_blocks.count_parameters(model1)
if config.pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(config.pretrained, map_location=device)
        model1.load_state_dict(checkpoint)
        #model2.load_state_dict(checkpoint)
        model1.eval()
        #model2.eval()

if True:
    model2=[]
    first=config.model_params['fc1_p']=[None,1]
    model2.append(model_(**config.model_params))
    second=config.model_params['fc1_p']=[None,5760]
    model2.append(model_(**config.model_params))



# configuration=make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True,"width":12, "height":12}, debug=True)
win_count_0=0
game_played=0


check_step=-2
visualizer = Visualizer()
model_player=0


act_space = act_spaces.__dict__[config.act_space] # .get_action_space()
obs_space = obs_spaces.__dict__[config.obs_space] # .get_obs_spec((12,12))
reward_space = reward_spaces.__dict__[config.reward_space]()  # .get_obs_spec((12,12))

configuration = {"seed": 1234567, "loglevel": 2, "annotations": True, "width": 12, "height": 12}
env = LuxEnv(obs_space=obs_space(), act_space=act_space(), reward_space=reward_space, configuration=configuration)

env.observation_space = env.obs_space.wrap_env(env=env.observation_space)
env = wrappers.LoggingEnv(wrappers.RewardSpaceWrapper(env, reward_space))

brain1=Brain_v1( discounting=0.9,lambda_w=0.4,lambda_t=0.8,alpha_w=0.002,alpha_t=0.01,model=model1)
brain2=Brain_v2( discounting=0.9,lambda_w=0.4,lambda_t=0.4,alpha_w=0.02,alpha_t=0.02,model=model2)
agent1 = agent_v2(player=0, action_space=act_space(default_board_dims=(12, 12)),environment=copy.deepcopy(env),brain=brain1,use_learning=True)

agent2 = agent_v2(player=1, action_space=act_space(default_board_dims=(12, 12)),environment=copy.deepcopy(env),brain=brain2,use_learning=True)

agents = [agent1, agent2]
episodes = 600
size = 1

logs,visulizer=None,None
for eps in range(episodes):
    for ag in agents:
        ag.brain.reset()
        ag.reset_environment(copy.deepcopy(env))
        ag.environment.add_logs_and_visulizer(logs, visulizer)


    print("=== Episode {} ===".format(eps))
    kaggle_env = make("lux_ai_2021",
               configuration=configuration,
               debug=True)
    steps = kaggle_env.run(agents)
    for ag in agents:
        ag.post_game(steps[-1][0]['observation'])
    logs,visulizer=agents[1].metric(eps)
    if (eps+1)%50==0:
        torch.save(model1.state_dict(), str(flags.savedir) + '/' +
               'rl_model1' + "_" + str(eps) + ".pth")



    print("++++++++++++completed episode {}+++++++++++++++".format(eps))

page=kaggle_env.render(mode="html")
with open(config.renderdir+'game.html', 'w') as f:
     f.write(page)

