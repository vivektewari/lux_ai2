import copy
import random

import torch
from kaggle_environments import make
from from_vivek import brain
import numpy as np
import models.conv_blocks as conv_blocks2
import models.baseline_models as conv_blocks
import models.combined_model as combined_models
from from_isaiah.lux_env import LuxEnv2
from from_isaiah.lux_gym.act_spaces import BasicActionSpace
from from_isaiah.lux_gym import obs_spaces, act_spaces, reward_spaces, wrappers
from from_lux.from_kaggle_env import __agent_runner
from kaggle_environments.agent import Agent
from from_vivek.brain import Brain_v1,Brain_v2
from from_vivek.env import LuxEnv
import time
from types import SimpleNamespace
import yaml
from tracking import agent_v2
from utils.visualizer import Visualizer
from collections import deque
from auxilary import visdom_print,VisdomLinePlotter
domains = [0 for i in range(10)]
my_queue = deque(domains)

np.random.seed(123456)
torch.manual_seed(123456)
    ###################################################
import os
#os.system("python3 -m visdom")
#os.system("python3 -m visdom.server")
with open('./config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))
flags = config
model_ = []
#chnage
model_.append(combined_models.__dict__[config.nn_model1_name])
model_.append(combined_models.__dict__[config.nn_model2_name])
if config.nn_model3_name is not None:
    model_.append(conv_blocks.__dict__[config.nn_model3_name])
brain_class = brain.__dict__[config.brain]
#conv_blocks.count_parameters(model1)
brain1=brain_class( discounting=0,lambda_w=0.2,lambda_t=0.90,alpha_w=0.05,alpha_t=0.005,model=None)
brain2=brain_class( discounting=0,lambda_w=0.2,lambda_t=0.90,alpha_w=0.05,alpha_t=0.005,model=None)

config.weightdir=config.rootdir+config.weightdir
track_loc_dir=config.rootdir+config.trackingdir
config.renderdir=config.rootdir+config.renderdir
model_params=[config.model_paramsfc1,config.model_paramsfc2]
if config.nn_model3_name is not None:
    model_params.append(config.model_paramsfc1)
if True:

    brain1.load_model(model_constructor=model_,model_params=model_params,dir=config.pretraindDir,identifier=1,eps=100
                      )
    brain2.load_model(model_constructor=model_, model_params=model_params,
                      dir=config.pretraindDir,identifier=2,eps=100)

# configuration=make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True,"width":12, "height":12}, debug=True)
win_count_0=0
game_played=0


check_step=-2
visualizer = Visualizer()
model_player=0


act_space = act_spaces.__dict__[config.act_space] # .get_action_space()
obs_space = obs_spaces.__dict__[config.obs_space] # .get_obs_spec((12,12))
reward_space = reward_spaces.__dict__[config.reward_space]()  # .get_obs_spec((12,12))

configuration = {"seed": 12345, "loglevel": 1, "annotations": True, "width": 12, "height": 12}
env = LuxEnv(obs_space=obs_space(), act_space=act_space(), reward_space=reward_space, configuration=configuration)

env.observation_space = env.obs_space.wrap_env(env=env.observation_space)
env = wrappers.LoggingEnv(wrappers.RewardSpaceWrapper(env, reward_space))


#track_loc_dir=None
agent1 = agent_v2(player=0, action_space=act_space(default_board_dims=(12, 12)),environment=copy.deepcopy(env),brain=brain1,use_learning=1,track_loc=track_loc_dir)

agent2 = agent_v2(player=1, action_space=act_space(default_board_dims=(12, 12)),environment=copy.deepcopy(env),brain=brain2,use_learning=1,track_loc=None)

agents = [agent1, agent2]
episodes = 5000#500#0000
size = 1

logged,visulizer=[None,None],VisdomLinePlotter(config.visdom_identifier)
start=time.time()
for eps in range(episodes):
    configuration = {"seed": 2370, "loglevel": 1, "annotations": True, "width": 12, "height": 12}#city_fuel_normalized
    for ag in agents:
        ag.brain.reset()
        ag.reset_environment(copy.deepcopy(env))
        ag.environment.add_logs_and_visulizer(logged[ag.id], visulizer)
        #ag.environment.env.reward_space.add_id(ag.id)



    print("=== Episode {} ===".format(eps))
    kaggle_env = make("lux_ai_2021",
               configuration=configuration,
               debug=True)
    steps = kaggle_env.run(agents)
    logged=[]
    for ag in agents:
        ag.post_game(steps[-1][0]['observation'])
        ag.environment.prepare_logs()
        logged.append(ag.environment.logs)
        ag.brain.optimize()

    visulizer=visdom_print(visulizer, eps,ag.environment.track_dict.keys(),logged[0],logged[1])
    print(time.time()-start)
    tracking_game=1
    if (eps+1)%tracking_game==0:
        agents[0].track_loc=track_loc_dir
    elif (eps)%tracking_game==0:
        agents[0].track_loc = None
    if (eps)%tracking_game==0:
        if 1:
            brain1.save_model(config.weightdir,eps,1)
            brain2.save_model(config.weightdir, eps,2)


        page = kaggle_env.render(mode="html")
        with open(config.renderdir + str(eps)+'game.html', 'w') as f:
            f.write(page)



    print("++++++++++++completed episode {}+++++++++++++++".format(eps))



