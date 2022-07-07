from dataLoaders import *
from catalyst.dl import SupervisedRunner, CallbackOrder, Callback, CheckpointCallback
from config import *
from auxilary import *
from funcs import get_dict_from_class, count_parameters

from losses import BCELoss
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from catalyst import dl
import math, sys
from from_lux.game import Game
from from_lux.game_map import Cell, RESOURCE_TYPES
from from_lux.constants import Constants
from from_lux.game_constants import GAME_CONSTANTS
from from_lux import annotate


DIRECTIONS = Constants.DIRECTIONS
game_state = None


def agent_vt(observation, configuration):
    """
    Algorithm:
    initialization for step 0>collects state from game_state_object> use fn:input state to get state in tensor>execute fn:model.predict to get
    get action values>choose episilon greedy action ftn:choose_action> update last state output value with
    q_learning from action> run optimization step for 1 epoch to change papram values ftn:model.fit> load currect state
    into last state> next iteration
    :param observation:
    :param configuration:
    :return: Actions in form of string list
    """
    global game_state,model,map_size,last_q_s_a,last_state_tensor,last_player_units,criterion,\
        optimizer,last_reward,my_city,opponent_city,visualizer,count

    ### Do not edit ###

    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.id
        map_size = observation['width']

        #getting model params and model|start
        if "model" not in  globals():
            print('new model created')
            model_param_ = get_dict_from_class(model_param)
            model_param_['start_channel'] =  state_dim_per_square
            model_param_['fc1_p'][-1] = (map_size ** 2) * action_count_per_square
            model = FeatureExtractor(**model_param_)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            count_parameters(model)
            if pretrained is not None :
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load(pretrained, map_location=device)
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                except:
                    model.load_state_dict(checkpoint)
                model.eval()
        # getting model params and model|stop
        # last state initialization
        last_q_s_a = None
        last_state_tensor=None
        last_player_units=None
        last_reward=(observation['reward']%10000)*10+100*(int(observation['reward']/10000))

    else:
        game_state._update(observation["updates"])



    state_tensor,player_units = input_state(game_state,observation)  # state tensor:1d player_units:unit_kkeper object
    state_tensor_temp=state_tensor.reshape((1,map_size, map_size, state_dim_per_square))
    state_tensor=torch.swapaxes(torch.swapaxes(state_tensor_temp, 2, 3), 1, 2)

    model_output= model(state_tensor) # output:1d
    model_output=model_output.reshape(map_size, map_size, action_count_per_square)
    # print(torch.max(model_output).detach(),torch.mean(model_output).detach(),torch.min(model_output).detach())

    q_s_a =  model_output# reshaping to position*actions
    actions,player_unit_dict=choose_action(action_value=q_s_a,player_units=player_units ,game_map=game_state.map,greedy_epsilon=greedy_episilon)
    if last_q_s_a is not None:#update the Q learning matrix
        current_reward=(observation['reward']%10000)*1+10*(int(observation['reward']/10000))
        #current_reward = observation['reward']
        reward=current_reward-last_reward

        for u in  last_player_units:
            if u.choosen_action_index !=-1:
                q_last_state=last_q_s_a[u.obj.pos.x,u.obj.pos.y,u.get_index()]
                if u.id in player_unit_dict.keys(): q_state=player_unit_dict[u.id].action_value
                else: q_state=extinct_value
                temporal_difference=reward+gamma*q_state-q_last_state
                q_last_changed = q_last_state + step_size*temporal_difference
                last_q_s_a[u.obj.pos.x,u.obj.pos.y, u.get_index()]=q_last_changed

        last_reward=current_reward
        last_q_s_a_temp=last_q_s_a.flatten().reshape((1,map_size*map_size*action_count_per_square))

        #model fitting prep and run
        model = perform_fit(model=model,x=last_state_tensor,y=last_q_s_a_temp,data_loader=data_loader,criterion=criterion,optimizer=optimizer)

    last_player_units=player_units
    last_state_tensor = state_tensor
    last_q_s_a = q_s_a

    return actions


if __name__ == "__main__":
    import torch.nn.functional as F
    import torch

    c = NeuralNet(sizes=[5, 12, 12], act_funcs=[F.relu_ for i in range(3)])
    d = c(torch.tensor([i for i in range(5)], dtype=torch.float32))
    #print(d.shape)
