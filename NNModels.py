import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):

  def __init__(self,inputSize,hiddenSize,outputSize,args):
    super().__init__()
    self.args=args.copy()
    self.linear1=nn.Linear(inputSize,hiddenSize)
    self.bn1=nn.BatchNorm1d(hiddenSize)
    self.dropOut=nn.Dropout(args['dropoutRate'])
    self.linear2=nn.Linear(hiddenSize,int(hiddenSize/2))
    self.bn2 = nn.BatchNorm1d(int(hiddenSize/2))
    self.head=nn.Linear(int(hiddenSize/2),outputSize)

    if self.args['manualWeightInit']:
      nn.init.xavier_uniform_( self.linear1.weight)
      nn.init.xavier_uniform_(self.linear2.weight)
      nn.init.xavier_uniform_(self.head.weight)

  def forward(self,input):
    out = input.view(input.size(0), -1)
    out=self.linear1(out)

    if self.args['batchNorm']:
       out=self.bn1(out)
    if self.args['applyDropOut']:
      out=self.dropOut(out)
    out=torch.relu(out)

    out=self.linear2(out)
    if self.args['batchNorm']:
       out=self.bn2(out)
    if self.args['applyDropOut']:
       out=self.dropOut(out)

    out=torch.relu(out)
    out=self.head(out)
    return  out


