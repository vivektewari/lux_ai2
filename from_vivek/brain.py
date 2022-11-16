from abc import ABC, abstractmethod
import torch,copy
import torch.nn as nn
class Brain(ABC):
    @abstractmethod
    def learn(self,rewards,v_t,v_t_plus_one)->None:
        pass

    @abstractmethod
    def reset(self) -> None:
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
    def note(self,v_t,action_prob):
        self.v_t=v_t
        self.action_prob=action_prob

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

    def learn(self,reward,v_t_plus_one) ->None:

        #delta = (max(reward,0) + self.discounting * v_t_plus_one - 0*self.v_t.detach()).detach()
        delta = reward #(reward+ self.discounting * v_t_plus_one - self.v_t.detach()).detach()#todo: correct it
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
                self.z_w[idx] =0*self.discounting * self.lambda_w * self.z_w[idx] +torch.clip(eval_gradients_base[idx],-1,1) #todo chage
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
        model_output=torch.cat((policy_logits,baseline),dim=0)
        return model_output
    def save_model(self,dir,eps,identifier):
        for i in range(2):
            pth='rl_duall_model' +str(i) +"player"+str(identifier)+"_"+ str(eps) + ".pth"
            torch.save(self.models[i].state_dict(), str(dir) + '/' +pth)

    def load_model(self,model_constructor:nn.Module,model_params:dict,identifier:int,eps:int,dir=None):
        self.models=[]
        model_params['fc1_p'][1] = 1
        first =copy.deepcopy(model_params)
        model_params['fc1_p'][1] =  5760
        second =copy.deepcopy(model_params)
        model_list=[first,second]
        for i in range(2):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_=model_constructor(**model_list[i])
            if dir is None:
                pass
            else:
                pth = 'rl_duall_model' + str(i) + "player" + str(identifier) + "_" + str(eps) + ".pth"
                checkpoint = torch.load(dir+pth, map_location=device)
                model_.load_state_dict(checkpoint)

            model_.eval()
            self.models.append(model_)


        # model2.load_state_dict(checkpoint)



