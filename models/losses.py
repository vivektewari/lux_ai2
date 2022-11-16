import torch.nn as nn
import numpy as np
import torch
EPSILON_FP16 = 1e-5
import torch.autograd as ag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import copy
class BCELoss(nn.Module):
    def __init__(self):

        super().__init__()
        self.func = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, actual):
        # bs, s, o = pred.shape
        # pred = pred.reshape(bs*s, o)
        #pred = self.sigmoid(pred)
        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)

        return self.func(pred, actual)

class LocalizatioLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bounding_loss = self.bounding_box_l2
        self.classification_loss =self.l2_loss# BCELoss()#
        self.bouding_conv_func= lambda x :x
    def l2_loss(self,pred, actual):
        #actual=torch.where(actual==1,1 ,0)
        loss =(torch.pow(torch.mean(torch.pow(1-pred[[i for i in range(len(pred))],[actual.tolist()]],2),dim=1),1/2))
        return loss[0]

    def bounding_box_l2(self,pred, actual):
        """
        :param pred:list of tuple in n dimension
        :param actual: list of tuple in n dimension
        :return: sum of l2 distance for each point
        """
        num_points = len(pred)
        loss = 0#torch.zeros(pred[0].shape)
        for i in range(num_points):
            loss +=torch.sum(torch.pow(torch.sum(torch.pow(pred[i]-actual[i],2),dim=1),1/2))

        return loss/(pred[0].shape[0]*2)

    def loss_calc(self, pred_classif, actual_classif,pred_bounding,actual_bounding):
        loss=self.bouding_conv_func(self.bounding_loss(pred_bounding,actual_bounding))+10*self.classification_loss(pred_classif, actual_classif)
        return loss
    def convert(self,x):
        x_len=x.shape[1]-4

        return x[:,0:x_len],[x[:,x_len:x_len+2],x[:,x_len+2:x_len+4]]
    def forward(self,pred,actual):
        pred_classif, pred_bounding=self.convert(pred)
        actual_classif, actual_bounding=self.convert(actual)
        return self.loss_calc( pred_classif, torch.tensor(actual_classif,dtype=torch.long)[:,0],pred_bounding,actual_bounding)
    def get_individual_loss(self,actual,pred):
        pred_classif, pred_bounding = self.convert(pred)
        actual_classif, actual_bounding = self.convert(actual)
        boundig_loss = self.bouding_conv_func(self.bounding_loss(pred_bounding,actual_bounding))
        classification_loss=self.classification_loss(pred_classif, torch.tensor(actual_classif,dtype=torch.long)[:,0])
        return boundig_loss , classification_loss


class with_1_step_grad():
    def __init__(self,discounting,lambda_w,lambda_t,alpha_w,alpha_t,model):
        self.discounting=discounting
        self.lambda_w=lambda_w
        self.lambda_t=lambda_t
        self.alpha_w = alpha_w
        self.alpha_t = alpha_t
        self.model=model
        self.opt=torch.optim.Adam(self.model.parameters(), lr=self.alpha_t)
        self.myzeros={}
        for idx, p in enumerate(self.model.parameters()):
            self.myzeros[idx] = torch.zeros(p.data.shape,requires_grad=True)

    def optimization_pass(self,reward,value_t_plus_one,value_t,z_w,z_t,I,action_prob,test=None):

        delta=(reward+self.discounting*value_t_plus_one-value_t).detach()
        #self.opt.zero_grad()
        eval_gradients_base= {}

        value_t.backward(retain_graph=True)
        for idx, p in enumerate(self.model.parameters()):
            eval_gradients_base[idx] = copy.deepcopy(p.grad)
            p.grad.zero_()

        eval_gradients_policy={}
        #eval_gradients_policy = ag.grad(torch.log(action_prob), self.model.parameters())

        s=torch.log(action_prob)
        s.backward()
        for idx, p in enumerate(self.model.parameters()):
             eval_gradients_policy[idx] = copy.deepcopy(p.grad)
             p.grad.zero_()
        #print(torch.max(eval_gradients_policy[1]), torch.max(eval_gradients_base[1]))
        #with torch.autograd.set_detect_anomaly(True):



        with torch.no_grad():
            for  idx,p in enumerate(self.model.parameters()):
                z_w[idx] = self.discounting * self.lambda_w* z_w[idx] + eval_gradients_base[idx]
                z_t[idx] = self.discounting * self.lambda_t* z_t[idx] + I*eval_gradients_policy[idx]
                #p.copy_(self.alpha_w*delta* z_w[idx]+self.alpha_t*delta* z_t[idx])
                temp= p.data+self.alpha_w*delta* z_w[idx]+self.alpha_t*delta* z_t[idx]
                p.copy_(temp.data)
        #print(torch.max(eval_gradients_policy[1]),torch.max(eval_gradients_base[1]))
        I=self.discounting*I


        return z_w,z_t,I,delta.item(),action_prob.detach()
