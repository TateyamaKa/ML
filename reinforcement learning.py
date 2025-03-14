import numpy as np
import random

class RL:

    def __init__(self,gamma=0.9,left_prob=0.1,right_prob=0.1,straight_prob=0.8,length=4,width=3,reward=[],max_iter=30,terminal=2):
        self.gamma = gamma
        self.left_prob = left_prob
        self.right_prob = right_prob
        self.straight_prob = straight_prob
        self.length = length
        self.width = width
        self.reward = reward
        self.value = np.zeros(length*width)
        self.policy = []
        self.max_iter = max_iter
        self.terminal = terminal
        self.check = (len(reward)==length*width)
    
    def prob(self):
        if self.check==1:
            labels = [0,1,-1]
            weight = [self.straight_prob,self.left_prob,self.right_prob]
            return random.choices(labels,weights=weight,k=1)[0]
        else:
            print('invailid initial setting.')
    
    def max_value(self,state,value):
        if self.check==1:
            if (state+1) % self.length!=0 and (state+1) % self.length!=1 and state+1>self.length and state+1<self.length*self.width-self.length+1:
                vup = self.straight_prob*value[state-self.length]+self.left_prob*value[state-1]+self.right_prob*value[state+1]
                vdown = self.straight_prob*value[state+self.length]+self.left_prob*value[state+1]+self.right_prob*value[state-1]
                vright = self.straight_prob*value[state+1]+self.left_prob*value[state-self.length]+self.right_prob*value[state+self.length]
                vleft = self.straight_prob*value[state-1]+self.left_prob*value[state+self.length]+self.right_prob*value[state-self.length]
            elif (state+1) % self.length==0 and state+1!=self.length and state+1!=self.length*self.width:
                vup = self.straight_prob*value[state-self.length]+self.left_prob*value[state-1]+self.right_prob*value[state]
                vdown = self.straight_prob*value[state+self.length]+self.left_prob*value[state]+self.right_prob*value[state-1]
                vright = self.straight_prob*value[state]+self.left_prob*value[state-self.length]+self.right_prob*value[state+self.length]
                vleft = self.straight_prob*value[state-1]+self.left_prob*value[state+self.length]+self.right_prob*value[state-self.length]
            elif (state+1) % self.length==1 and state+1!=1 and state+1!=self.length*self.width-self.length+1:
                vup = self.straight_prob*value[state-self.length]+self.left_prob*value[state]+self.right_prob*value[state]
                vdown = self.straight_prob*value[state+self.length]+self.left_prob*value[state+1]+self.right_prob*value[state]
                vright = self.straight_prob*value[state+1]+self.left_prob*value[state-self.length]+self.right_prob*value[state+self.length]
                vleft = self.straight_prob*value[state]+self.left_prob*value[state+self.length]+self.right_prob*value[state-self.length]
            elif state+1>1 and state+1<self.length:
                vup = self.straight_prob*value[state]+self.left_prob*value[state-1]+self.right_prob*value[state+1]
                vdown = self.straight_prob*value[state+self.length]+self.left_prob*value[state+1]+self.right_prob*value[state-1]
                vright = self.straight_prob*value[state+1]+self.left_prob*value[state]+self.right_prob*value[state+self.length]
                vleft = self.straight_prob*value[state-1]+self.left_prob*value[state+self.length]+self.right_prob*value[state]
            elif state+1 >self.length*self.width-self.length+1 and state+1<self.width*self.length:
                vup = self.straight_prob*value[state-self.length]+self.left_prob*value[state-1]+self.right_prob*value[state+1]
                vdown = self.straight_prob*value[state]+self.left_prob*value[state+1]+self.right_prob*value[state-1]
                vright = self.straight_prob*value[state+1]+self.left_prob*value[state-self.length]+self.right_prob*value[state]
                vleft = self.straight_prob*value[state-1]+self.left_prob*value[state]+self.right_prob*value[state-self.length]
            elif state==0:
                vup = self.straight_prob*value[state]+self.left_prob*value[state]+self.right_prob*value[state+1]
                vdown = self.straight_prob*value[state+self.length]+self.left_prob*value[state+1]+self.right_prob*value[state]
                vright = self.straight_prob*value[state+1]+self.left_prob*value[state]+self.right_prob*value[state+self.length]
                vleft = self.straight_prob*value[state]+self.left_prob*value[state+self.length]+self.right_prob*value[state]
            elif state==self.length-1:
                vup = self.straight_prob*value[state]+self.left_prob*value[state-1]+self.right_prob*value[state]
                vdown = self.straight_prob*value[state+self.length]+self.left_prob*value[state]+self.right_prob*value[state-1]
                vright = self.straight_prob*value[state]+self.left_prob*value[state]+self.right_prob*value[state+self.length]
                vleft = self.straight_prob*value[state-1]+self.left_prob*value[state+self.length]+self.right_prob*value[state]
            elif state==self.length*self.width-1:
                vup = self.straight_prob*value[state-self.length]+self.left_prob*value[state-1]+self.right_prob*value[state]
                vdown = self.straight_prob*value[state]+self.left_prob*value[state]+self.right_prob*value[state-1]
                vright = self.straight_prob*value[state]+self.left_prob*value[state-self.length]+self.right_prob*value[state]
                vleft = self.straight_prob*value[state-1]+self.left_prob*value[state]+self.right_prob*value[state-self.length]
            elif state==self.length*self.width-self.length:
                vup = self.straight_prob*value[state-self.length]+self.left_prob*value[state-1]+self.right_prob*value[state]
                vdown = self.straight_prob*value[state]+self.left_prob*value[state+1]+self.right_prob*value[state]
                vright = self.straight_prob*value[state+1]+self.left_prob*value[state-self.length]+self.right_prob*value[state]
                vleft = self.straight_prob*value[state]+self.left_prob*value[state]+self.right_prob*value[state-self.length]
            vs = [vup,vdown,vleft,vright]
            maxv =  max(vs)
            maxindex = vs.index(maxv)
            return [maxv,maxindex]
        else:
            print('invailid initial setting.')
    
    def value_iteration(self):
        if self.check==1:
            for _ in range(self.max_iter):
                s=0
                temp = self.value.copy()
                for i in range(self.length*self.width):
                    if i == self.terminal:
                        self.value[i] = self.reward[s]
                    else:
                        self.value[i] = self.reward[s]+self.gamma*self.max_value(s,temp)[0]
                    s += 1
        else:
            print('invailid initial setting.')

    
    def get_policy(self):
        if self.check==1:
            for s in range(self.width*self.length):
                if s==self.terminal:
                    self.policy.append('T')
                else:
                    if self.max_value(s,self.value)[1]==0:
                        self.policy.append('↑')
                    elif self.max_value(s,self.value)[1]==1:
                        self.policy.append('↓')
                    elif self.max_value(s,self.value)[1]==2:
                        self.policy.append('←')
                    elif self.max_value(s,self.value)[1]==3:
                        self.policy.append('→')
            po = np.array(self.policy)
            return po.reshape(self.width,self.length)
        else:
            print('invailid initial setting.')
    
    def get_payoff(self):
        if self.check == 1:
            pay = self.value
            return pay.reshape(self.width,self.length)
        else:
            print('invailid initial setting.')



rl = RL(gamma=0.9,left_prob=0.1,right_prob=0.1,straight_prob=0.8,length=3,width=2,reward=[-0.1, -1, 1, -0.1, -0.1, -0.1],max_iter=50,terminal=2)
rl.value_iteration()
print(rl.get_policy())
print(rl.get_payoff())