import torch
from model import Net
#################
##  Code Here  ##
#################


# global variables declaration
#--------------------------------------------------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
#--------------------------------------------------------------------------------------------------


# model instantiation
#--------------------------------------------------------------------------------------------------
model = Net().to(device)
#--------------------------------------------------------------------------------------------------

  
# Print model information
#--------------------------------------------------------------------------------------------------
#################
##  Code Here  ##
#################
#--------------------------------------------------------------------------------------------------