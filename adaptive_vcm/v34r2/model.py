"""Separate GT risk, source risk, feasibility and measured rate predictions."""
import torch
from torch import nn
import torch.nn.functional as F
from ..v34.model import Controller as R1Controller


class Controller(R1Controller):
    def __init__(self,actions=13,conditions=9):
        super().__init__(actions,conditions)
        self.head=nn.Sequential(nn.Linear(24+conditions,32),nn.SiLU(),nn.Linear(32,actions*9))

    def forward(self,x,maps,condition):
        if x.ndim!=5 or maps.shape!=x.shape or condition.shape!=(len(x),self.conditions): raise ValueError('invalid R2 controller input')
        b,t,_,h,w=x.shape
        z=self.encoder(torch.cat((x,maps),2).reshape(b*t,6,h,w)).reshape(b,t,16)
        _,hidden=self.temporal(z)
        raw=self.head(torch.cat((hidden[-1],condition),1)).reshape(b,self.actions,9)
        return dict(logits=raw[:,:,0],gt_risk=raw[:,:,1:4],source_risk=raw[:,:,4:7],
                    rate=F.softplus(raw[:,:,7])+.05,feasible_logits=raw[:,:,8])
