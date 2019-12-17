import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import math

import pandas_datareader.data as web

style.use('ggplot')
matplotlib.use( 'tkagg' )




class Pricing(object):
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
        df.to_csv('profit.csv')
    def _sales_model(self,price):

        #Hypothese: linear price sales function

        menge = self.cfg.LINEAR_A -self.cfg.LINEAR_B * price
        return menge


    def _price_model(self):
        '''
        caculate the price with
        :return: price
        '''
        df=self.df
        price = (1 + df['Aufschlagsatz'])*df['Stückkosten']  # 5.1
        if self.cfg.OPT_FUNTION =='linear':
            # optimal price in linear price sale function  menge = a -bp
            opt_price = 1/2*(self.cfg.LINEAR_A/self.cfg.LINEAR_B + df['Stückkosten'] )  # 5.7
        elif self.cfg.OPT_FUNTION == 'gutenburg' :
            opt_price = self.cfg.GUTENBERG_A - self.cfg.GUTENBERG_C1* \
                        math.sinh(self.cfg.GUTENBERG_C2*(price-self.cfg.COM_AVERAGE_PRICE))  #5.10
        elif self.cfg.OPT_FUNTION =='oligopol_linear': # P_ = Alpha + beta *price   #q = a-b*p+c*p_
            opt_price = 1/2 *((self.cfg.LINEAR_A+ self.cfg.LINEAR_C* self.cfg.COM_LINEAR_ALPHA)\
                              /(self.cfg.LINEAR_B-self.cfg.LINEAR_C*self.cfg.COM_LINEAR_BETA)+df['Stückkosten'])


        return price, opt_price

    def reaktions_hypothese(self, price, com_price):
        '''

        :param price: price of the product
        :param com_price: price of the competition (other company)
        :return: new price
        '''
        if self.cfg.REAKTION_HYPOTHESE == 'cournot':
            new_price = price
        elif self.cfg.REAKTION_HYPOTHESE == 'chamberlin':
            new_price = com_price
        else :
            _, new_price =  self._price_model()  ##according to the change B, to calculate a new optimal price

        return new_price


    def _cost_model(self):
        menge = self._sales_model()
        cost = f(menge)  #todo  define the funtion between cost and menge
        return cost

    def profit_model(self,df):
        '''
        caculate the profit with the funtion 5.4

        :param df:
        :return:
        '''
        price = self._price_model()
        menge = self._sales_model()
        cost = self._cost_model()
        profit= price*menge - cost

        return profit

    def mengenbezogenen_preisdifferenzierung(self, menge, price):
        '''
        :param menge: the sales menge
        :param price: the price of sales
        :return: the profit
        '''

        if self.cfg.TARIF == 'zeiteilig':
            profit = self.cfg.FIXED_GRUNDGKOST + menge * price
        elif self.cfg.TARIF == 'block':
            if menge >= self.cfg.TARIF_QB:
                profit = price * menge
            else:
                profit = self.cfg.FIXED_GRUNDGKOST + price * self.cfg.TARIF_DISCOUNT*menge
        elif self.cfg.TARIF == 'angestoßen':
            if menge >= self.cfg.TARIF_QB:
                profit = price* menge
            else:
                profit = menge*price*self.cfg.TARIF_DISCOUNT


    def price_elasticity(self):
        pass


    def break_even_point(self, df):
        '''
        caculate the break even menge
        :param df: muss include lable ['Preis','Stückkosten','c_fix']
        :return: break even menge
        '''
        p = df['Preis']
        k = df['Stückkosten']
        d = p -k  #Stückdeckungsbeitrag
        c_fix= df['Fixkosten']
        q_be = c_fix/d  # break even Menge

        return q_be

    def customer(self):
        pass




