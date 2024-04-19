
import torch
import torch.nn as nn
import numpy as np

class DeepFittingNet(nn.Module):
    def __init__(self,input_size=3,hidden_size=32,num_layers=8,nonlinearity='relu', bidirectional=True,batch_first=True,
                 output_dim=2,device='cuda:0'):
        super(DeepFittingNet, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity = nonlinearity,
                            bidirectional=bidirectional, batch_first=batch_first)
        self.iOut=3  #original
        #self.iOut = 6
        if bidirectional:
            # fcInputChannel =2*num_layers*hidden_size
            fcInputChannel = 2*self.iOut*hidden_size
            h0 = torch.zeros((2*num_layers,1,hidden_size)).to(device)
        else:
            # fcInputChannel =  num_layers * hidden_size
            fcInputChannel = self.iOut*hidden_size
            h0 = torch.zeros((num_layers,1,hidden_size)).to(device)

        # learnable hidden states
        self.h0 = nn.Parameter(h0, requires_grad=True)

        # fuly connected prediction layer
        self.fc1 = nn.Sequential(
        nn.Linear(in_features= fcInputChannel, out_features=400),
        nn.LeakyReLU(),
        nn.Linear(in_features=400, out_features=400),
        nn.LeakyReLU(),
        nn.Linear(in_features=400, out_features=200),
        nn.LeakyReLU(),
        nn.Linear(in_features=200, out_features=200),
        nn.LeakyReLU(),
        nn.Linear(in_features=200, out_features=100),
        nn.LeakyReLU(),
        nn.Linear(in_features=100, out_features=50),
        nn.LeakyReLU(),
        nn.Linear(in_features=50, out_features=30),
        nn.LeakyReLU(),
        nn.Linear(in_features=30, out_features=output_dim),
        )

        self.fc2 = nn.Sequential(
        nn.Linear(in_features= fcInputChannel, out_features=400),
        nn.LeakyReLU(),
        nn.Linear(in_features=400, out_features=400),
        nn.LeakyReLU(),
        nn.Linear(in_features=400, out_features=200),
        nn.LeakyReLU(),
        nn.Linear(in_features=200, out_features=200),
        nn.LeakyReLU(),
        nn.Linear(in_features=200, out_features=100),
        nn.LeakyReLU(),
        nn.Linear(in_features=100, out_features=50),
        nn.LeakyReLU(),
        )

        self.fc3 =  nn.Sequential(
        nn.Linear(in_features=51, out_features=30),
        nn.LeakyReLU(),
        nn.Linear(in_features=30, out_features=output_dim),
        )


    def residualCal(self,x,ABTx):
        Ti = x[:, :, 1]
        inputSig = x[:, :, 0]
        sigLen = x.shape[1]
        Aarr = ABTx[:, 0].repeat(sigLen).reshape([sigLen, ABTx.shape[0]])
        Aarr = torch.transpose(Aarr, 0, 1)

        Barr = ABTx[:, 1].repeat(sigLen).reshape([sigLen, ABTx.shape[0]])
        Barr = torch.transpose(Barr, 0, 1)
        Tx = torch.abs(ABTx[:, 2])+torch.ones(ABTx.shape[0],1).to('cuda:0')*0.00000001
        Txarr = torch.abs(Tx[:, 2].repeat(sigLen).reshape([sigLen, Tx.shape[0]]))
        Txarr = torch.transpose(Txarr, 0, 1)

        resSigSq = torch.square(torch.abs(Aarr + Barr * torch.exp(-Ti/Txarr) )- inputSig)
        residualSum = torch.mean(resSigSq, dim=1).reshape([ABTx.shape[0], 1])

        return residualSum


    def forward(self, x):

        nb, nt, ch = x.shape
        output, hn = self.rnn(x, (self.h0.repeat((1, nb, 1))))
        nt, nb, ch = hn.shape
        Ti = x[:, :, 1]
        inputSig = x[:, :, 0]

        ABTx = self.fc1(output[:, -self.iOut:, :].reshape((nb, output.shape[2] * self.iOut)))

        return ABTx
