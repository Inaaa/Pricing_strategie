import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')
matplotlib.use( 'tkagg' )

##download data titel  ### the name of the input.


##load data and save it.
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

        menge = cfg.LINEAR_A -cfg.LINEAR_B * price
        return menge


    def customer(self):
        pass
    def _price_model(self):
        '''
        caculate the price with
        :return: price
        '''
        df=self.df
        price = (1 + df['Aufschlagsatz'])*df['Stückkosten']  # 5.1
        # optimal price in linear price sale function  menge = a -bp
        opt_price = 1/2*(cfg.LINEAR_A/cfg.LINEAR_B + df['Stückkosten'] )  # 5.7

        return price, opt_price

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



        pass

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

        retuen q_be



