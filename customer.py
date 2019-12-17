import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import math

import pandas_datareader.data as web

import torch
import torch.nn.functional as F


style.use('ggplot')
matplotlib.use( 'tkagg' )




class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

class Customer_net(object):
    def __init__(self,name,cfg):
        '''

        :param name: name of input .csv file
        '''
        self.name = name
        self.df = pd.read_csv(self.name,parse_datas=True,index_col=0)
        self.cfg = cfg
    def choose_data(self):
        '''
        combine the data from the many .csv files
        :return:
        '''

        df=self.df
        df.set_index('Date', inplace=True)

        df.drop(['**', '**', '**', '**'], 1, inplace=True)##keep price , sales, profit
        print(main_df.head())
        df.to_csv('customer.csv')
    def predict_paramter(self):
        net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
        #print(net)  # net architecture

        optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

        plt.ion()

        for t in range(100):
            prediction = net(x)  # input x and predict based on x

            loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if t % 10 == 0:
                # plot and show learning process
                plt.cla()
                plt.scatter(x.data.numpy(), y.data.numpy())
                plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
                plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
                plt.show()
                plt.pause(0.1)

        plt.ioff()
        torch.save(net, 'net.pkl')

    def restore_net(self,x):
        # restore entire net to net2
        net2 = torch.load('net.pkl')
        prediction = net2(x)
