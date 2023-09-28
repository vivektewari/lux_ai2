from abc import ABC, abstractmethod
import torch,copy
import torch.nn as nn
import numpy as np
from codes.models.losses import custom_mape_loss
from codes.from_vivek.brain_aux import *
from codes.utils.gradient_metric import grad_information
class Brain(ABC):
    @abstractmethod
    def learn(self,rewards,v_t,v_t_plus_one)->None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def model(self) -> torch.tensor:
        pass

    def add_model(self, model) -> None:
        self.model=model
class Brain_v1(Brain):
    def __init__(self,discounting,lambda_w,lambda_t,alpha_w,alpha_t,model):
        self.discounting=discounting
        self.lambda_w=lambda_w
        self.lambda_t=lambda_t
        self.alpha_w = alpha_w
        self.alpha_t = alpha_t
        self.models=model
        # self.opt=torch.optim.Adam(self.model.parameters(), lr=self.alpha_t)
        # self.myzeros={}
        # for idx, p in enumerate(self.model.parameters()):
        #     self.myzeros[idx] = torch.zeros(p.data.shape,requires_grad=True)
    def reset(self) -> None:
        self.z_w,self.z_t,self.I={},{},1.0

        for idx, p in enumerate(self.models.parameters()):
            self.z_w[idx] = torch.zeros(p.data.shape)
            self.z_t[idx] = torch.zeros(p.data.shape)
    def note(self,v_t,action_prob,for_learner,model_input):
        self.v_t=v_t
        self.action_prob=action_prob
        self.for_learner=for_learner
        self.model_input=model_input

    def learn(self,reward,v_t_plus_one) ->None:
        delta = (reward + self.discounting * v_t_plus_one - self.v_t.detach()).detach()
        # self.opt.zero_grad()
        eval_gradients_base = {}

        self.v_t.backward(retain_graph=True)
        for idx, p in enumerate(self.models.parameters()):
            eval_gradients_base[idx] = copy.deepcopy(p.grad)
            p.grad.zero_()

        eval_gradients_policy = {}
        # eval_gradients_policy = ag.grad(torch.log(action_prob), self.model.parameters())

        s = torch.log(self.action_prob)
        s.backward()
        for idx, p in enumerate(self.models.parameters()):
            eval_gradients_policy[idx] = copy.deepcopy(p.grad)
            p.grad.zero_()
        # print(torch.max(eval_gradients_policy[1]), torch.max(eval_gradients_base[1]))
        # with torch.autograd.set_detect_anomaly(True):

        with torch.no_grad():
            for idx, p in enumerate(self.models.parameters()):
                self.z_w[idx] = self.discounting * self.lambda_w * self.z_w[idx] + eval_gradients_base[idx]
                self.z_t[idx] = self.discounting * self.lambda_t * self.z_t[idx] + self.I * eval_gradients_policy[idx]
                # p.copy_(self.alpha_w*delta* z_w[idx]+self.alpha_t*delta* z_t[idx])
                temp = p.data + self.alpha_w * delta * self.z_w[idx] + self.alpha_t * delta * self.z_t[idx]
                p.copy_(temp.data)
        # print(torch.max(eval_gradients_policy[1]),torch.max(eval_gradients_base[1]))
        self.I = self.discounting * self.I
        return delta
    def model(self,input):
        return self.models(input)
class Brain_v2(Brain_v1):#different model for actor and critic


    def learn(self,reward,v_t_plus_one=None) ->None:

        #delta = (max(reward,0) + self.discounting * v_t_plus_one - 0*self.v_t.detach()).detach()
        if v_t_plus_one is None:v_t_plus_one=torch.tensor(0.0)
        delta = (reward+ self.discounting * v_t_plus_one - self.v_t.detach()).detach()
        #if self.action_prob.detach() < 0.000000000000001 : return delta#or reward == 0
        # self.opt.zero_grad()
        eval_gradients_base = {}

        self.v_t.backward(retain_graph=True)
        for idx, p in enumerate(self.models[0].parameters()):
            eval_gradients_base[idx] = copy.deepcopy(p.grad)
            p.grad.zero_()

        eval_gradients_policy = {}
        # eval_gradients_policy = ag.grad(torch.log(action_prob), self.model.parameters())


        s = torch.log(self.action_prob)
        s.backward()
        for idx, p in enumerate(self.models[1].parameters()):
            eval_gradients_policy[idx] = copy.deepcopy(p.grad)
            p.grad.zero_()
        # print(torch.max(eval_gradients_policy[1]), torch.max(eval_gradients_base[1]))
        # with torch.autograd.set_detect_anomaly(True):

        with torch.no_grad():
            for idx, p in enumerate(self.models[0].parameters()):
                self.z_w[idx] =self.discounting * self.lambda_w * self.z_w[idx] +torch.clip(eval_gradients_base[idx],-0.1,0.1) #todo chage
                # p.copy_(self.alpha_w*delta* z_w[idx]+self.alpha_t*delta* z_t[idx])
                update = p.data + self.alpha_w * delta * self.z_w[idx]
                if not (torch.isinf(update).any() and torch.isnan(update).any()):
                    p.copy_(update)
            #delta = reward
            # if sum(eval_gradients_policy[6]) >0.000000001 and reward>0:
            #     d = 0
            oldq_data=[]
            for idx, q in enumerate(self.models[1].parameters()):
                self.z_t[idx] =0*self.discounting * self.lambda_t * self.z_t[idx] +  self.I * torch.clip(eval_gradients_policy[idx],-0.1,0.1)
                # p.copy_(self.alpha_w*delta* z_w[idx]+self.alpha_t*delta* z_t[idx])
                update=q.data  +self.alpha_t * delta * self.z_t[idx]
                oldq_data.append(copy.deepcopy(q.data))
                if not (torch.isinf(update).any() or torch.isnan(update).any()):
                    q.copy_(update)

                else :
                    print(('nan faced in paramaeter update {}'.format(idx)))

        #print(torch.max(eval_gradients_policy[1]),torch.max(eval_gradients_base[1]))

        self.I = self.discounting * self.I# keep it one else gradient will become too small for later steps
        return delta
    def reset(self) -> None:
        self.z_w,self.z_t,self.I={},{},1.0

        for idx, p in enumerate(self.models[0].parameters()):
            self.z_w[idx] = torch.zeros(p.data.shape)
        for idx, p in enumerate(self.models[1].parameters()):
            self.z_t[idx] = torch.zeros(p.data.shape)
    def model(self,input):
        baseline=self.models[0](input)
        policy_logits=self.models[1](input)
        #model_output=torch.cat((policy_logits,baseline),dim=0)
        model_output = [policy_logits, baseline]
        return model_output
    def save_model(self,dir,eps,identifier):
        for i in range(2):
            pth='rl_duall_model' +str(i) +"player"+str(identifier)+"_"+ str(eps) + ".pth"
            torch.save(self.models[i].state_dict(), str(dir) + '/' +pth)

    def load_model(self,model_constructor:nn.ModuleList,model_params:dict,identifier:int,eps:int,dir=None):
        self.models=[]
        #model_params['fc1_p'][1] = 1
        first =copy.deepcopy(model_params[0])
        #model_params['fc1_p'][1] =  5760
        second =copy.deepcopy(model_params[1])
        model_list=[first,second]
        n_models=2
        if len(model_params)==3:
            n_models=3
            common = copy.deepcopy(model_params[2])
            model_list = [first, second,common]
            model_comb=model_constructor[2](**model_list[2])

        for i in range(2):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_=model_constructor[i](**model_list[i])
            if n_models==3:
                #model_list[i]['premodel']=model_comb
                #model_=model_constructor[i](**model_list[i])
                model_.add_premodel(model_comb)


            if dir is None:
                pass
            else:
                pth = 'rl_duall_model' + str(i) + "player" + str(identifier) + "_" + str(eps) + ".pth"
                checkpoint = torch.load(dir+pth, map_location=device)
                model_.load_state_dict(checkpoint)

                model_.eval()
            self.models.append(model_)


        # model2.load_state_dict(checkpoint)

class Brain_v3(Brain_v2):#using loss function instea of self gradient correction
    def reset(self) -> None:

        self.z_w, self.z_t, self.I = {}, {}, 1.0
        self.pred_values=[]
        self.entropies=[]
        self.rewards=[]
        self.optimizer=[torch.optim.SGD(self.models[0].parameters(), lr=self.alpha_w,weight_decay=0),torch.optim.SGD(self.models[1].parameters(), lr=self.alpha_t,weight_decay=0.01)]
        self.loss=nn.MSELoss()#nn.L1Loss(reduction='mean')#nn.L1Loss()#custom_mape_loss()#
    def learn2(self,reward,v_t_plus_one=None) ->None:

        self.pred_values.append(self.v_t)
        self.rewards.append(reward)
        if v_t_plus_one is not None:v_t_plus_one=torch.tensor(0.)
        else:
            if 1:

                self.optimizer[0].zero_grad()
                #baseline function increment
                pred_values=torch.stack(self.pred_values)

                targets=td_lambda(torch.tensor(self.rewards),pred_values.flatten(),torch.tensor(0),discounts=torch.ones((len(self.rewards)))*self.discounting,lmb=1)
                target_values=targets[0]
                loss_bs=self.loss(pred_values.flatten(),target_values.detach().float())
                loss_bs.backward(retain_graph=True)
                #todo comment below lines
                grads=grad_information.get_grads(self.models[0])
                loop=0
                #print(self.models[0].fc2.weight[0, 5184], self.models[0].fc2.bias)
                for g in grads:
                    print(loop,np.max(g),np.min(g))
                    loop+=1

                torch.nn.utils.clip_grad_norm_(self.models[0].parameters(), max_norm=10)

                self.optimizer[0].step()

                dict={'bs_loss':loss_bs.detach(),'entropy':np.array(self.entropies).mean()}
                return dict

        if 1:
            #if True: return 1
            #print('wer')
            self.optimizer[1].zero_grad()
            #delta = (reward+ self.discounting * v_t_plus_one - self.v_t).detach()
            delta=reward*10

            s = -torch.log(self.action_prob)*delta
            #print(s.detach())
            #adding entropy loss
            cont=0
            for e in self.for_learner:
                if e[2]==2:
                    temp=e[3][e[4]]
                    scaled=temp#*(1/temp.sum())
                    part_entropy=-(scaled*torch.log(scaled)).mean()
                    if cont==0:entropy=part_entropy
                    else: entropy+=part_entropy
                    cont+=1
            if cont>0 :
                s+=-1*0.01*entropy/cont
                self.entropies.append(entropy.detach() / cont)
            s.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.models[1].parameters(), max_norm=1)
            self.optimizer[1].step()

            return 1
class Brain_v4(Brain_v3):
    #iterates of value if value converges to some limits or number of chnages
    #to policy iteration mode
    def __init__(self,discounting,lambda_w,lambda_t,alpha_w,alpha_t,model):
        super().__init__(discounting,lambda_w,lambda_t,alpha_w,alpha_t,model)
        self.game_number=0
        self.mode=0 #valuation
        self.model_run=0
        self.model_input_games = []
        self.target_values_games = []

    def reset(self):
        super().reset()
        #self.optimizer = [torch.optim.SGD(self.models[0].parameters(), lr=self.alpha_w, weight_decay=0),
        #                  torch.optim.SGD(self.models[1].parameters(), lr=self.alpha_t, weight_decay=0.01)]
        self.model_run+=1
        self.game_number+=1
        #todo:remove force mode
        self.mode=0 #self.mode_selection()
    def mode_selection(self):
        if self.mode==0 and self.model_run>4 :
            new_mode= 1
            print("switching to policy iteration")
        elif self.mode==1 and self.model_run>2:
            new_mode=0
            print("switching to value iteration")
        else: new_mode=self.mode
        if new_mode!= self.mode:
            self.model_run=0
            return new_mode
        else :return self.mode

    def learn2(self, reward, v_t_plus_one=None) ->None:
        dict=0
        self.pred_values.append(self.v_t)
        self.rewards.append(reward)
        if v_t_plus_one is not None:
            v_t_plus_one = torch.tensor(0.)
        else:
            if self.learn_value:
                self.optimizer[0].zero_grad()
                # baseline function increment
                pred_values = torch.stack(self.pred_values)
                discounts=torch.tensor([self.discounting for i in range(len(self.rewards),0,-1)])
                targets = td_lambda(torch.tensor(self.rewards), pred_values.flatten(), torch.tensor(0),
                                    discounts=discounts, lmb=1)
                target_values = targets[0]
                loss_bs = self.loss(torch.tanh(pred_values.flatten()), target_values.detach().float())
                if self.mode==0:loss_bs.backward(retain_graph=True)
                # todo comment below lines

                #print(self.models[0].fc2.weight[0, 5184], self.models[0].fc2.bias)
                #gradient watcher
                grads = grad_information.get_grads(self.models[0])
                loop = 0
                for g in grads:
                    print(loop, np.max(g), np.min(g))
                    loop += 1

                torch.nn.utils.clip_grad_norm_(self.models[0].parameters(), max_norm=10)

                if self.mode==0:self.optimizer[0].step()

                dict = {'bs_loss': loss_bs.detach(), 'entropy': np.array(self.entropies).mean(),'target_values':target_values.detach().float()}
                return dict

        if self.learn_policy:
            # if True: return 1
            # print('wer')
            self.optimizer[1].zero_grad()
            delta = (reward+ self.discounting * v_t_plus_one - self.v_t).detach()
            #delta = reward * 10

            s = -torch.log(self.action_prob) * delta
            # print(s.detach())
            # adding entropy loss
            cont = 0
            for e in self.for_learner:
                if e[2] == 2:
                    temp = e[3][e[4]]
                    scaled = temp  # *(1/temp.sum())
                    part_entropy = -(scaled * torch.log(scaled)).mean()
                    if cont == 0:
                        entropy = part_entropy
                    else:
                        entropy += part_entropy
                    cont += 1
            if cont > 0:
                s += -1 * 0.01 * entropy / cont
                self.entropies.append(entropy.detach() / cont)
            if self.mode==1:
                s.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.models[1].parameters(), max_norm=10)

                eval_gradients_policy={}
                eval_gradients_policy2 = {}
                if 0:
                    for idx, p in enumerate(self.models[1].premodel.parameters()):

                        eval_gradients_policy[idx] = copy.deepcopy(p.data)
                        p.grad.zero_()
                self.optimizer[1].step()
                if 0:
                    for idx, p in enumerate(self.models[1].premodel.parameters()):
                        eval_gradients_policy2[idx] = copy.deepcopy(p.data)
                        p.grad.zero_()
                    print(torch.mean(torch.abs(eval_gradients_policy[0] - eval_gradients_policy2[0])))
                u=0

        return dict
    def optimize(self):
        if (self.game_number) % 30 == 0:

            loss_minmization = []
            for i in range(50):
                self.optimizer[0].zero_grad()
                output = torch.cat(
                    [self.model(self.model_input_games[i])[1] for i in range(len(self.model_input_games))])
                loss_bs = self.loss(output.flatten(), torch.cat(self.target_values_games).float())
                loss_minmization.append(loss_bs.detach())
                loss_bs.backward(retain_graph=True)
                self.optimizer[0].step()
            #print(loss_minmization)
            self.model_input_games = []
            self.target_values_games = []

    def learn(self, reward, v_t_plus_one=None) ->None: #collect 20 game value and then set for optimization
        dict=0
        self.pred_values.append(self.v_t)

        self.model_input_games.append(self.model_input)
        self.rewards.append(reward)
        if v_t_plus_one is not None:
            v_t_plus_one = torch.tensor(0.)
        else:
            if self.learn_value:

                # baseline function increment
                pred_values = torch.stack(self.pred_values)
                discounts=torch.tensor([self.discounting for i in range(len(self.rewards),0,-1)])
                targets = td_lambda(torch.tensor(self.rewards), pred_values.flatten(), torch.tensor(0),
                                    discounts=discounts, lmb=1)
                target_values = targets[0]

                #self.pred_values_games.append(torch.tanh(pred_values.flatten()))
                self.target_values_games.append(target_values)
                loss_bs = 10*self.loss(pred_values.flatten(), target_values.detach().float())
                if self.mode==1:loss_bs.backward(retain_graph=True)
                # todo comment below lines

                #print(self.models[0].fc2.weight[0, 5184], self.models[0].fc2.bias)
                #gradient watcher
                # grads = grad_information.get_grads(self.models[0])
                # loop = 0
                # for g in grads:
                #     print(loop, np.max(g), np.min(g))
                #     loop += 1
                #
                # torch.nn.utils.clip_grad_norm_(self.models[0].parameters(), max_norm=10)

                if self.mode==1:self.optimizer[0].step()

                dict = {'bs_loss': loss_bs.detach(), 'entropy': np.array(self.entropies).mean(),'target_values':target_values.detach().float()}
                return dict

        if self.learn_policy:
            # if True: return 1
            # print('wer')
            self.optimizer[1].zero_grad()
            delta = (reward+ self.discounting * v_t_plus_one - self.v_t).detach()
            #delta = reward * 10

            s = -torch.log(self.action_prob) * delta
            # print(s.detach())
            # adding entropy loss
            cont = 0
            for e in self.for_learner:
                if e[2] == 2:
                    temp = e[3][e[4]]
                    scaled = temp  # *(1/temp.sum())
                    part_entropy = -(scaled * torch.log(scaled)).mean()
                    if cont == 0:
                        entropy = part_entropy
                    else:
                        entropy += part_entropy
                    cont += 1
            if cont > 0:
                s += -1 * 0.01 * entropy / cont
                self.entropies.append(entropy.detach() / cont)
            if self.mode==1:
                s.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.models[1].parameters(), max_norm=10)

                eval_gradients_policy={}
                eval_gradients_policy2 = {}
                if 0:
                    for idx, p in enumerate(self.models[1].premodel.parameters()):

                        eval_gradients_policy[idx] = copy.deepcopy(p.data)
                        p.grad.zero_()
                self.optimizer[1].step()
                if 0:
                    for idx, p in enumerate(self.models[1].premodel.parameters()):
                        eval_gradients_policy2[idx] = copy.deepcopy(p.data)
                        p.grad.zero_()
                    print(torch.mean(torch.abs(eval_gradients_policy[0] - eval_gradients_policy2[0])))
                u=0

        return dict




