#Imports
from robin_stocks.robinhood.account import build_holdings
from robin_stocks.robinhood.stocks import find_instrument_data, get_instruments_by_symbols, get_latest_price,\
get_stock_historicals
from robin_stocks.robinhood.orders import order_buy_market
from robin_stocks.robinhood.authentication import login
from matplotlib import pyplot as plt
import random


class TradeException (Exception):
        def __init__(self,msg):
            super().__init__(msg)


class DataPoint:
    def __init__(self,symol):
        self.symbol =
        self.price_history

class Trader:
    def __init__(self):
        self.ownership = {}
        self.session = None
        pass

    def retrieve_stock_info(self,symbol):
        return get_latest_price(symbol)

    def make_trade(symbol,type='shares',qty=0):
        if type == 'shares':
            pass
        try:
            order_buy_market(symbol,10)
        except Exception as e:
            print(e)

    def login(self,user,password,expires=3600):
        self.session = login(user,password,expiresIn=expires)
        self.stocks = build_holdings()

    def history(self,name):
        get_stock_historicals()


    def get_full_dataset(self):
        pass

    def graph_history(self,names,interval='hour',span='day'):
        data = get_stock_historicals(names,interval=interval,span=span)
        x_vals = []
        y_vals = []
        print(data)
        input()
        for timeStamp in r:
            x_vals.append(timeStamp['begins_at'])
            y_vals.append(float(timeStamp['high_price']))
        plt.scatter(x_vals,y_vals,s=1)
        plt.show()



current_trader = Trader()
current_trader.login("everetts","NAVYlaptop52!!",expires=24*60*60)
current_trader.graph_history(["aapl","sphd"],)
