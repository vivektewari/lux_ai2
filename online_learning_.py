import torch

from monobeast_lux_ai import create_env
from models.losses import with_1_step_grad
import numpy as np
import models.conv_blocks as conv_blocks
#from from_isaiah.lux_env import LuxEnv2
from from_vivek.env import LuxEnv
from from_isaiah.lux_gym.act_spaces import BasicActionSpace
from from_isaiah.lux_gym import obs_spaces, act_spaces, reward_spaces, wrappers
from from_lux.from_kaggle_env import __agent_runner
from kaggle_environments.agent import Agent
from types import SimpleNamespace
import yaml
from vivek_agent import agent_v1
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
model2 = model_(**config.model_params)
conv_blocks.count_parameters(model1)
if config.pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(config.pretrained, map_location=device)
        model1.load_state_dict(checkpoint)
        model1.eval()

# configuration=make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True,"width":12, "height":12}, debug=True)
win_count_0=0
game_played=0
agent1 = agent_v1(player=0,action_space=BasicActionSpace(default_board_dims=(12,12)))
agent1.add_model(model1)
agent2 = agent_v1(player=1,action_space=BasicActionSpace(default_board_dims=(12,12)))
agent2.add_model(model2)
change_maker=with_1_step_grad(discounting=0.9,lambda_w=0.4,lambda_t=0.4,alpha_w=0.2,alpha_t=0.1,model=model1)
change_maker2=with_1_step_grad(discounting=0.9,lambda_w=0.4,lambda_t=0.4,alpha_w=0.2,alpha_t=0.1,model=model2)

check_step=-2
visualizer = Visualizer()
model_player=0
for i in range(1000):
    model2.init_weight()


    act_space = act_spaces.__dict__[config.act_space]()  # .get_action_space()
    obs_space = obs_spaces.__dict__[config.obs_space]()  # .get_obs_spec((12,12))
    reward_space = reward_spaces.__dict__[config.reward_space]()  # .get_obs_spec((12,12))

    configuration = {"seed": 1234567, "loglevel": 1, "annotations": True, "width": 12, "height": 12}
    env = LuxEnv(obs_space=obs_space, act_space=act_space, reward_space=reward_space, configuration=configuration)
    #env.add_agents([lambda y:['m u2 n'],lambda y:['m u2 n']])

    agents = [agent1, agent2]
    for ag in agents: ag.action_space =act_space
    #runner = __agent_runner(env.sub_environment, agents)



    env.observation_space = env.obs_space.wrap_env(env.observation_space)
    env = wrappers.LoggingEnv(wrappers.RewardSpaceWrapper(env, reward_space))
    obs=env.reset()


    output1=runner[0].act(obs[0],env.game_state,obs[3]['remainingOverageTime'][0])
    output2 = runner[1].act(obs[0], env.game_state, obs[3]['remainingOverageTime'][1])
    start=True
    step = 0
    count=0
    z_w,z_t,test={},{},{}
    z_w2, z_t2, test2 = {}, {}, {}
    I = 1.0
    I2=1.0
    for idx, p in enumerate(model2.parameters()):
        z_w[idx] = torch.zeros(p.data.shape)
        z_t[idx] = torch.zeros(p.data.shape)
        z_w2[idx] = torch.zeros(p.data.shape)
        z_t2[idx] = torch.zeros(p.data.shape)


    loss,losses,probs=0,0,0
    while not env.sub_environment.done:


            output1 = runner[0].act(obs[0], env.game_state, obs[3]['remainingOverageTime'][0])
            output2 = runner[1].act(obs[0], env.game_state, obs[3]['remainingOverageTime'][1])
            outputs=[output1,output2]
            try:
                output = [output1[0]['action_for_env'], output2[0]['action_for_env']]

                #print(output)
                #print(output)
            except:
                t=0


            if i>1 and step==check_step:
                c=0
                print(2, (agent1(obs[0],1, env.game_state)['action_for_env'] == oldag1))
                print(4, (output1[0]['baseline'] -baseline))
            if step==check_step:
                old_obs = obs[0]
                oldoutput=output
                old_model=agent1.model.parameters()


                oldag1=agent1(obs[0],1, env.game_state)['action_for_env']
                baseline=output1[0]['baseline']
            step+=1
            obs = env.step(output)
            #prob=outputs[model_player+0][0]['policy_logits'].detach()
            #loss=outputs[model_player+0][0]['policy_logits'].detach()
            value_t_plus_one=agents[model_player](obs[0], 1, env.game_state)['baseline']
            z_w,z_t,I,loss,prob=change_maker.optimization_pass(reward=obs[1][model_player],value_t_plus_one=value_t_plus_one,value_t=outputs[model_player][0]['baseline'],z_w=z_w,z_t=z_t,I=I,action_prob=outputs[model_player][0]['policy_logits'],test=test)
            # # value_t_plus_one = agents[model_player+1](obs[0], 1, env.game_state)['baseline']
            # z_w2, z_t2, I2, loss2, prob2 = change_maker2.optimization_pass(reward=obs[1][model_player+1],
            #                                                          value_t_plus_one=value_t_plus_one,
            #                                                          value_t=outputs[model_player+1][0]['baseline'],
            #                                                          z_w=z_w2, z_t=z_t2, I=I2,
            #                                                          action_prob=outputs[model_player+1][0][
            #                                                              'policy_logits'], test=test)

            #print(step)
            losses+=abs(loss)
            probs+=prob



            if env.sub_environment.done:
                winner=int(env.sub_environment.state[0]['reward']>env.sub_environment.state[1]['reward'])
                win_count_0 =win_count_0+winner
                game_played +=1
                my_queue.pop()
                my_queue.appendleft(winner)
                aver=np.array(my_queue).mean()
                print(win_count_0/game_played)
                if aver>1.8:
                    agent2.add_model(model1)
                    agent1.add_model(model2)
                    print('agent2 takes')
                    model_player=1
                elif aver<-0.1:
                    agent2.add_model(model2)
                    agent1.add_model(model1)
                    model_player = 0
                    print('agent1 takes')

                visualizer.display_current_results(i, aver,
                                                   name='win0')
                visualizer.display_current_results(i,losses/step,
                                                   name='losses')
                visualizer.display_current_results(i, probs/step,
                                                   name='probs')
                # value_t_plus_one.alpha_t=(1-aver)*10
                # value_t_plus_one.alpha_w =(1 - aver)*10

                if i%50==0:

                    torch.save(model1.state_dict(), str(flags.savedir) + '/' +
                                                  'rl_model'+ "_" +str(i) + ".pth")


                break
# x = env.sub_environment.render(mode='html')
# f = open('/home/pooja/PycharmProjects/lux_ai/outputs/GFG1.html', 'w')
# f.write(x)
# f.close()
