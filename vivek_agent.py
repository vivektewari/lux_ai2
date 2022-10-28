
import numpy as np
import pandas as pd
from torch.nn.functional import softmax,sigmoid
from abc import ABC, abstractmethod
from from_isaiah.lux.game import  Game
from from_lux.game_objects import Unit, CityTile,City
from from_isaiah.act_spaces import get_unit_action,get_city_tile_action
from torch import nn
from from_isaiah.utils import flags_to_namespace, Stopwatch
from from_isaiah.utils import DEBUG_MESSAGE, RUNTIME_DEBUG_MESSAGE, LOCAL_EVAL
import torch
import gym
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union,List
from from_isaiah.lux_gym.reward_spaces import RewardSpec
class abstract_agent(ABC):
    def __init__(self,action_space=None,player=0,board_dim=(12,12),model_updater=None,environment=None,track_loc=None):

        self.id=player
        self.board_dim=board_dim
        self.stopwatch = Stopwatch()
        self.model_updater=model_updater
        self.environment=environment
        self.act_space = environment.action_space
        self.obs_space = environment.obs_space
        self.track_loc=track_loc


        pass

    @abstractmethod
    def add_model(self,model,path=None):
         pass

    def tracking(self, output,key):
        return None
    @abstractmethod
    def see(self,obs)->np.ndarray:

        """
        interface between environment output and model

        :param obs:
        :return:
        """
        pass
    #@abstractmethod
    def model_update(self):
        pass
    #@abstractmethod
    def speech(self,model_output:np.ndarray)->Dict[str, np.ndarray]:
        """
        this is interfacing agent to act spaces
        :param model_output: np.ndarray
        :return: dict space
        """
        pass




    def __call__(self,obs):#obs from lux ai engine
        self.stopwatch.reset()
        self.stopwatch.start("Observation processing")
        #self.extract_update_game(self, obs)

        eye=self.see(obs)
        self.stopwatch.stop().start("model running")
        model_output=self.model(eye)
        self.stopwatch.stop().start("model output processing")
        agent_output=self.speech(model_output)
        self.stopwatch.stop()
        if agent_output is None:
            u=0
        return agent_output

    def preprocess(self, game_state) -> NoReturn:
        """
        for move:Getting set of cool down 0 positions,opposing city position:set position
        for transfer:Giver as non zero resource,taker has space,taker is allied
        build city: if have required resources,tile is not resource tile:allied sets
        constraints
        :param game_state:
        :return:dict[str->self/opp_set:dict2]|dict2[str->position|contraints:object,name,actionable|constrint count]
        """

        no_move=set([])

        constraint=0
        pos_to_city_tile_dict={}
        pos_to_unit_dict={}
        is_nyt=game_state.is_night
        for player in game_state.players:
            for city in player.cities.values(): #list [city,units|workers]
                for city_tile in city.citytiles:
                    pos_to_city_tile_dict[city_tile.pos.astuple()]=city_tile
                    if city_tile.team!=self.id:no_move.add(city_tile.pos.astuple())
                    else:constraint+=1
                if city.fuel<city.light_upkeep and is_nyt:
                    if city.team==self.id:constraint-=len(city.citytiles) # correcting contraintremoving all citues if they are going to die next turn
            for unit in player.units:
                if not unit.can_act():no_move.add(unit.pos.astuple())
                pos_to_unit_dict[unit.pos.astuple()] = unit
                if unit.team==self.id:constraint-=1

        masks=self.act_space.get_action_mask(game_state=game_state,pos_to_unit_dict=pos_to_unit_dict,pos_to_city_tile_dict=pos_to_city_tile_dict,no_move=no_move,my_id=self.id,constraint=constraint)
        return masks,pos_to_city_tile_dict,pos_to_unit_dict



        # Remove data augmentations if there are fewer overage seconds than 2x the number of data augmentations


class agent_v1(abstract_agent):
    def see(self,obs):
        return torch.tensor(obs,dtype=torch.float32)#np.swapaxes(obs,1,3)
    def add_model(self,model,path=None):
        self.model=model
    def format_model_output(self,model_output:torch.tensor)->Dict[str,Union[torch.tensor,int]]:## 12,12,[10,8,4]

        dict = {}
        dict['baseline'] = torch.tensor(model_output[-1], requires_grad=True)
        temp = model_output[:-1].reshape((12, 12, 40))  # .reating_grph()
        # temp=temp.clone().detach().requires_grad_(True)
        dict['unit'], dict['cart'], dict['ct'] = torch.tensor(temp[:, :, :19], requires_grad=True), torch.tensor(
            temp[:, :, 19:36], requires_grad=True), torch.tensor(temp[:, :, 36:40], requires_grad=True)
        return dict



    def action_selection(self,game:Game,model_output_:dict)->Dict[str,torch.tensor]:# dict witk key:policy logits ->shape x,y,type,logit |, action->x,y,type,action

        masks,pos_to_city_tile_dict,pos_to_unit_dict=self.preprocess(game)

        baseline=model_output_[-1] #torch.sigmoid(

        model_output = model_output_[:-1].reshape((12, 12, 40))
        #for tracking:
        model_output_1=torch.softmax(model_output[:,:,0:19].detach(),dim=2),torch.softmax(model_output[:,:,19:36].detach(),dim=2) \
                                    ,torch.softmax(model_output[:,:,36:40].detach(),dim=2)
        noaction_probs=torch.cat((model_output_1[0][:,:,0].reshape((12, 12, 1))
                                  ,model_output_1[1][:,:,0].reshape((12, 12, 1)),model_output_1[2][:,:,0].reshape((12, 12, 1))),dim=2)

        selected_action=-1
        action=[]
        logits=1
        action_str=[]


        reference=['ct','unit','cart']

        #in case multiple unints on same city the only 1 moves. This is to choosen so that 1 spatial spae occupies one unit
        cities=list(pos_to_city_tile_dict.keys())
        units=list(pos_to_unit_dict.keys())
        loop=0
        for key in cities+units:
            loop+=1
            if loop<=len(cities):
                ct=pos_to_city_tile_dict[key]
                if ct.team != self.id: continue
                action_set = model_output[key[0],key[1],36:40]
                mask=masks[ct.cityid][(ct.pos.x,ct.pos.y)]

                id=ct.cityid
                ref= 1
                noaction_probs[key][2]=torch.nan
            else:
                #continue #TODO
                unit=pos_to_unit_dict[key]
                if unit.team != self.id: continue
                mask=masks[unit.id]
                id=unit.id
                if unit.is_worker():
                    action_set =model_output[key[0],key[1],:19]
                    ref=2
                    noaction_probs[key][0] = torch.nan
                else:
                    action_set = model_output[key[0],key[1],19:36]
                    ref=3
                    noaction_probs[key][1] = torch.nan


            if sum(mask)==1:continue
            else:

                relevant_action_set=action_set.clone().detach()
                max_,min_=torch.max(relevant_action_set),torch.min(relevant_action_set)
                #print(relevant_action_set)
                scalar=max([max_/30,-min_/30,1])

                # doing for gradient intactness
                #relevant_action_set[mask==0]=-1
                remaining_prob=torch.exp(relevant_action_set[mask==1]/scalar).sum()
            choices=torch.tensor([i for i in range(len(relevant_action_set))])[mask==1]

            distribution=softmax(relevant_action_set[choices])
            if self.use_learning and np.random.random()<50/self.game_played:selected_action=np.random.choice(choices)#torch.argmax(relevant_action_set)#,p=np.array(distribution)
            else:selected_action=np.random.choice(choices,p=np.array(distribution))
            #if selected_action==3 and ref==1:print(relevant_action_set[choices])
            action.append(torch.tensor([key[0], key[1],ref, selected_action]))
            if ref==1:
                self.record_value_for_log(
                    {'city_action_research':int(selected_action==3),'unit_build':int(selected_action==1),'city_no_action':int(selected_action==0),'city_no_action_prob':distribution[0]})

            elif ref==2:
                self.record_value_for_log( {'bcity': int(selected_action==2),'unit_no_action':int(selected_action==0),'move_action':selected_action in (3,4,5,6),'unit_no_action_prob':distribution[0]})


            if loop <= len(cities): act_str=get_city_tile_action(ct,selected_action)
            else:act_str=get_unit_action(unit,selected_action,pos_to_unit_dict)
            if act_str is not None:
                action_str.append(act_str)

            probs = torch.exp(action_set[selected_action]/scalar)/(remaining_prob)
            if torch.isnan(probs) or torch.isinf(probs):
                check=2

            logits*=probs

        self.record_value_for_log({'blank_space_no_act_prob':torch.nanmean(noaction_probs)})

        if selected_action ==-1:
            logits=torch.tensor(1.0,dtype=torch.float32,requires_grad=True)
            action=[torch.tensor(-1)]

        dict={'policy_logits':logits,'actions':torch.stack(action),'baseline':baseline,'action_for_env':action_str}
        if len(dict['action_for_env'])>0:
            if dict['action_for_env'][0].find('bw')>=0:
                d=0
        return dict

    def __call__(self,vector:torch.tensor,obs,game:Game)->Dict[str,torch.tensor]:#policy_logits,baseline,action:12,12,3,1
        temp=self.see(vector)
        temp=self.model(temp)
        #temp=self.format_model_output(temp)
        dict=self.action_selection(game,temp)
        return dict

class agent_v2(agent_v1):
    def __init__(self, action_space=None, player=0, board_dim=(12, 12), model_updater=None, environment=None,brain=None,use_learning=True,track_loc=None):

        self.id = player
        self.board_dim = board_dim
        self.stopwatch = Stopwatch()
        self.model_updater = model_updater
        self.environment = environment
        self.act_space = environment.action_space
        self.obs_space = environment.obs_space.wrap_env(environment.observation_space)
        self.game_step=0
        self.brain=brain
        self.use_learning=use_learning
        self.game_played=1
        self.track_loc=track_loc


        pass
    def extract_update_game(self,observation):
        if observation["step"] == 0:
            self.environment.initialize_game(observation)
            self.game_step=0

        else:
            self.environment.update_game(observation)
        self.game_step+=1
        #print(self.game_step)
    def see(self):
        #if self.game_step!=1:
            #self.update_model
        obs,reward,done,info = self.environment.get_obs_reward_done_info() #converting in vector form

        return torch.tensor(obs,dtype=torch.float32),reward[self.id],done

    def tracking(self, output,key):
        """
        save the snapshot of different part of game for reassceing it later
        """



        if key =='see':
            _, c, l, b = output.shape
            dump=pd.DataFrame(columns=[i for i in range(l)],index=[i for i in range(b)])
#['board_size', 'cart', 'cart_COUNT', 'cart_cargo_coal', 'cart_cargo_full', 'cart_cargo_uranium',
# 'cart_cargo_wood', 'cart_cooldown', 'city_tile', 'city_tile_cooldown', 'city_tile_cost', 'city_tile_fuel',
# 'coal', 'day_night_cycle', 'dist_from_center_x', 'dist_from_center_y', 'night', 'phase', 'research_points',
# 'researched_coal', 'researched_uranium', 'road_level', 'turn', 'uranium', 'wood', 'worker', 'worker_COUNT',
# 'worker_cargo_coal', 'worker_cargo_full', 'worker_cargo_uranium', 'worker_cargo_wood', 'worker_cooldown']
            if self.game_step == 1:
                self.writer = pd.ExcelWriter(self.track_loc+'see.xlsx')
                map=['board_size', 'cart', 'cart_COUNT', 'cart_cargo_coal', 'cart_cargo_full', 'cart_cargo_uranium',
                'cart_cargo_wood', 'cart_cooldown', 'city_tile', 'city_tile_cooldown', 'city_tile_cost', 'city_tile_fuel',
                'coal', 'day_night_cycle', 'dist_from_center_x', 'dist_from_center_y', 'night', 'phase', 'research_points',
                'researched_coal', 'researched_uranium', 'road_level', 'turn', 'uranium', 'wood', 'worker', 'worker_COUNT',
                'worker_cargo_coal', 'worker_cargo_full', 'worker_cargo_uranium', 'worker_cargo_wood', 'worker_cooldown']
                self.new_map=[]
                for i in range(len(map)):
                    self.new_map.append(map[i])
                    if map[i] in ['cart', 'cart_COUNT', 'cart_cargo_coal', 'cart_cargo_full', 'cart_cargo_uranium',
                'cart_cargo_wood', 'cart_cooldown', 'city_tile', 'city_tile_cooldown', 'city_tile_cost', 'city_tile_fuel',
                'research_points','researched_coal', 'researched_uranium','worker', 'worker_COUNT',
                'worker_cargo_coal', 'worker_cargo_full', 'worker_cargo_uranium', 'worker_cargo_wood', 'worker_cooldown']:
                        self.new_map.append(map[i]+'1')


            next_entry='board_size'
            for i in range(l):
                for j in range(b):
                    output_slice=list(output[:, :, i, j].flatten().numpy())

                    dump[i][j]={}
                    for k in range(c):
                        if output_slice[k]>0:dump[i][j][self.new_map[k]]=output_slice[k]

            #dump.to_csv(self.track_loc+str(self.game_step)+'_see.csv')



            dump.to_excel(self.writer, sheet_name=str(self.game_step))
            self.writer.save()


    def __call__(self,obs,configuration):#obs from lux ai engine
        """
        Making agent so that it can fit with lux ai engine to produce render output.
        for this agent needs game state, updation through obs(have) | code extract_with _update
        Then it need function transform the obs, reward to serve learning goal|see. rest model and interpretation is already in agent_v1
        :param obs: Lux ai obs
        :param configuration: lux ai cof
        :return:
        """
        self.stopwatch.reset()
        self.stopwatch.start("Observation processing")
        self.extract_update_game(obs)

        eye_output,reward,done=self.see()
        if self.track_loc is not None:self.tracking(eye_output,key='see')
        self.stopwatch.stop().start("model running")
        model_output=self.brain.model(eye_output)
        self.stopwatch.stop().start("model output processing")
        dict = self.action_selection(self.environment.game_state, model_output)


        if self.game_step == 1:
            self.brain.reset()
            self.brain.note(v_t=dict['baseline'],action_prob=dict['policy_logits'])
            self.last_notes=dict['actions'],eye_output,model_output[:-1].reshape(12,12,40),dict['policy_logits'],len(dict['actions']),dict['action_for_env']
        elif self.environment.done:
            value = self.brain.v_t.detach()
            delta = dict['baseline'].detach() + reward - value

            self.record_value_for_log(
                {'value': self.brain.v_t.detach(), 'delta': delta, 'action_prob': self.brain.action_prob.detach() })
            if self.use_learning:self.brain.learn(reward=reward, v_t_plus_one=torch.tensor(0.0))
            self.game_played+=1
        else:
            value = self.brain.v_t.detach()
            delta =self.brain.discounting*dict['baseline'].detach()+reward-value
            #recording value for logs
            #print(self.brain.action_prob.detach() )
            self.record_value_for_log({'value':self.brain.v_t.detach(),'delta':delta,'action_prob':self.brain.action_prob.detach() \
                                       })

            # self.environment.value.append(self.brain.v_t.detach())
            # self.environment.action_prob.append(self.brain.action_prob.detach())
            # self.environment.delta.append(delta)
            if self.use_learning:self.brain.learn(reward=reward,v_t_plus_one=dict['baseline'].detach())
            model_output = self.brain.model(eye_output)
            dict = self.action_selection(self.environment.game_state, model_output)
            self.brain.note(v_t=dict['baseline'],action_prob=dict['policy_logits'])
            # if self.id==0:
            #     print(self.last_notes[5])
            #     print(self.last_notes[0])
            #
            #print(self.last_notes[0][2], self.last_notes[0][3])
            #if self.id==0:print(self.environment.game_state.players[0].research_points)
            if False and self.last_notes[4]==1 and self.last_notes[5]!=[] and self.id==0:
                #self.last_notes[0]=torch.tensor([self.last_notes[0]])
                if self.last_notes[0][0][2]==1:
                    action_s=self.last_notes[0][0][3]+36
                    if self.last_notes[0][0][3]==3:


                    # elif self.last_notes[0][2]==3:action_s=self.last_notes[0][3]+19
                    # else :action_s=self.last_notes[0][3]
                        ref = self.last_notes[0][0][0], self.last_notes[0][0][1], action_s
                        c=self.brain.model(self.last_notes[1])[:-1].reshape((12, 12, 40))[ref]-self.last_notes[2][ref]
                        #d=self.brain.model(self.last_notes[1])[-1]-value
                        print(reward,self.last_notes[5])
                        print("plogits_change {}".format(c>0))#*delta
                    #print("base_valuation_change {}".format(d ))

            #except:l=1
            self.last_notes = dict['actions'], eye_output, model_output[:-1].reshape(12, 12, 40), dict[
                'policy_logits'], len(dict['actions']),dict['action_for_env']


        #self.game_step+=1
        return dict['action_for_env']
    def reset_environment(self,env):
        self.environment=env

    def record_value_for_log(self,dict):
        """
        Pass value for visulizaton
        :return:
        """
        for key in dict.keys():
            self.environment.track_dict[key].append(dict[key])


    def post_game(self,observation):
        #self.extract_update_game(self,observation)
        env=self.environment
        while hasattr(env, 'unwrapped'):
            setattr(env, 'done', True)
            if type(env) == type(env.unwrapped):
                break
            else:
                env=env.unwrapped
        self(observation,0)
    # def metric(self,eps):
    #
    #     self.environment.visdom_print(eps)
    #     return self.environment.logs,self.environment.visualizer












if __name__=='__main__':
    from monobeast_lux_ai import create_env
    import models.conv_blocks as conv_blocks
    from from_isaiah.lux_env import LuxEnv2
    from from_isaiah.lux_gym.act_spaces import BasicActionSpace
    from from_isaiah.lux_gym import obs_spaces, act_spaces, reward_spaces, wrappers
    from from_lux.from_kaggle_env import __agent_runner
    from kaggle_environments.agent import Agent
    from types import SimpleNamespace
    import yaml




    ###################################################

    with open('config.yaml', 'r') as f:
        config = SimpleNamespace(**yaml.safe_load(f))
    flags = config
    model_ = conv_blocks.__dict__[config.nn_model_name]
    model = model_(**config.model_params)
    conv_blocks.count_parameters(model)
    agent1 = agent_v1(player=0)
    agent1.add_model(model)
    agent2 = agent_v1(player=1)
    agent2.add_model(model)


    act_space = act_spaces.__dict__[config.act_space]()  # .get_action_space()
    obs_space = obs_spaces.__dict__[config.obs_space]()  # .get_obs_spec((12,12))
    reward_space = reward_spaces.__dict__[config.reward_space]()  # .get_obs_spec((12,12))

    # configuration=make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True,"width":12, "height":12}, debug=True)

    configuration = {"seed": 1234567, "loglevel": 1, "annotations": True, "width": 12, "height": 12}
    env = LuxEnv2(obs_space=obs_space, act_space=act_space, reward_space=reward_space, configuration=configuration)
    # env.add_agents([lambda y:['m u2 n'],lambda y:['m u2 n']])

    agents = [agent1, agent2]
    for ag in agents: ag.action_space =act_space
    runner = __agent_runner(env.sub_environment, agents)

    env.observation_space = env.obs_space.wrap_env(env.observation_space)
    env = wrappers.LoggingEnv(wrappers.RewardSpaceWrapper(env, reward_space))
    obs=env.reset()
    output1=runner[0].act(obs[0],env.game_state,obs[3]['remainingOverageTime'][0])
    output2 = runner[1].act(obs[0], env.game_state, obs[3]['remainingOverageTime'][1])

    while not env.sub_environment.done:


            if env.sub_environment.done: break
            output1 = runner[0].act(obs[0], env.game_state, obs[3]['remainingOverageTime'][0])
            output2 = runner[1].act(obs[0], env.game_state, obs[3]['remainingOverageTime'][1])

            output = [output1[0]['action_for_env'], output2[0]['action_for_env']]
            # except:
            #     t=0


            obs = env.step(output)
            print(obs[1])

