#Imports
from robin_stocks.robinhood.account import build_holdings
from robin_stocks.robinhood.stocks import find_instrument_data, get_instruments_by_symbols, get_latest_price,\
get_stock_historicals
from robin_stocks.robinhood.orders import order_buy_market
from robin_stocks.robinhood.authentication import login
from matplotlib import pyplot as plt
import random
from robin_stocks.tda.stocks import search_instruments
import yfinance 
import os
import time 


class TradeException (Exception):
        def __init__(self,msg):
            super().__init__(msg)


class DataPoint:

    fields = [
        'sector',
        'fullTimeEmployees',
        'preMarketPrice',
        'regularMarketPrice',
        'dayHigh',
        'bid',
        'dividendYield',
        'fiftyTwoWeekLow',
        'fiftyTwoWeekHigh',
        'fiveYearAvgDividendYield',
        'volume',
        'askSize',
        'ask',
        'dayLow',
        'averageVolume',
        'marketCap',
        'regularMarketVolume',
        'trailingPE',
        'regularMarketDayLow',
        'exDividendDate',
        'dividendRate',
        'averageVolume10days',
        'regularMarketPreviousClose',
        'averageDailyVolume10Day',
        'regularMarketDayHigh',
        'payoutRatio',
        'twoHundredDayAverage',
        'regularMarketOpen',
        'previousClose',
        'sharesShortPriorMonth',
        'shortPercentOfFloat',
        'forwardPE',
        'pegRatio',
        'dateShortInterest',
        'priceToSalesTrailing12Months',
        'earningsQuarterlyGrowth',
        'lastDividendDate',
        'priceHint',
        'enterpriseValue',
        'beta',
        'floatShares',
        'sharesShortPreviousMonthDate',
        'shortRatio',
        'mostRecentQuarter',
        'heldPercentInsiders',
        'priceToBook',
        'SandP52WeekChange',
        'lastDividendValue',
        'trailingEps',
        'netIncomeToCommon',
        'heldPercentInstitutions',
        'lastFiscalYearEnd',
        'sharesPercentSharesOut',
        'sharesShort',
        'bookValue',
        'sharesOutstanding',
        'forwardEps',
        '52WeekChange',
        'enterpriseToEbitda',
        'enterpriseToRevenue',
        'recommendationMean',
        'quickRatio',
        'revenuePerShare',
        'totalCashPerShare',
        'totalRevenue',
        'totalDebt',
        'totalCash',
        'targetHighPrice',
        'returnOnEquity',
        'debtToEquity',
        'targetMeanPrice',
        'numberOfAnalystOpinions',
        'returnOnAssets',
        'currentRatio',
        'earningsGrowth',
        'currentPrice',
        'targetMedianPrice',
        'freeCashflow',
        'grossProfits',
        'targetLowPrice',
        'ebitda',
        'operatingMargins',
        'revenueGrowth',
        'operatingCashflow',
        'grossMargins',
        'profitMargins',
        'ebitdaMargins',
        'maxAge']

    def __init__(self,symbol):
        self.symbol = symbol
        self.data_container = {}
    
    def refresh_stock_obj(self):
        self.data = yfinance.Ticker(self.symbol)

    def collect_fields(self):
        self.refresh_stock_obj()
        time_key = time.time()
        self.data_container[time_key] = {}
        for field in DataPoint.fields:
            try:
                self.data_container[time_key][field] = self.data.info[field]
            except KeyError:
                print(f"BAD KEY: {field}")
    
    def write_results(self):
        if not os.path.isdir("data"):
            os.mkdir("data")
        i = 0
        while os.path.isfile(rf"data\{self.symbol}{i}.txt"):
            i += 1
        file = open(rf"data\{self.symbol}{i}.txt","w")

        file.write(f"{','.join(DataPoint.fields)}\n")
        for tk in self.data_container:
            line = f"{tk}"
            for field in self.data_container[tk]:
                line += f",{self.data_container[tk][field]}"
            line += '\n'
            file.write(line)
        file.close()




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
    
    def get_info(self):
        print(search_instruments("AAPL","fundamental"))

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

class DataHandler:
    def __init__():
        pass
class StockHandler:
    def __init__():
        pass

current_trader = Trader()
current_trader.login("everettperson@gmail.com","NAVYlaptop52!!",expires=24*60*60)
dp = DataPoint("AAPL")
dp.collect_fields()
dp.write_results()