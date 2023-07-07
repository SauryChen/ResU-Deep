import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, 150)
        self.hidden2 = nn.Linear(150, 90)
        self.hidden3 = nn.Linear(90, 60)
        self.predict = nn.Linear(60, n_output)
    def forward(self, input):
        out = F.tanh(self.hidden1(input))
        out = F.tanh(self.hidden2(out))
        out = F.tanh(self.hidden3(out))
        out = F.sigmoid(self.predict(out))
        
        return out

model = Net(33, 1)
print(model)
