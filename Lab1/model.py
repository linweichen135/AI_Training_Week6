import torch
from torch import nn


# models definition
#--------------------------------------------------------------------------------------------------
def conv_bn_relu(inc, outc, stride):
    return nn.Sequential(
        nn.Conv2d(inc, outc, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(num_features=outc),
        nn.ReLU6(inplace=True)
    )
    
def convdw_bn_relu(inc, outc, stride):
    return nn.Sequential(
        nn.Conv2d(inc, inc, kernel_size=3, stride=stride, padding=1, groups=inc, bias=False),
        nn.BatchNorm2d(num_features=inc),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=outc),
        nn.ReLU6(inplace=True)
    )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #################
        ##  Code Here  ##
        #################
    def forward(self, x):
        #################
        ##  Code Here  ##
        #################
#--------------------------------------------------------------------------------------------------