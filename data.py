import yfinance as y
import pandas
from pprint import pp

class DataPoint:
    def __init__(self,symbol):
        self.stock = y.Ticker(symbol)
        self.price_history = None

    def pull_data(self,p='1mo',i='30m',include=10):
        self.dataFrame = self.stock.history(period='max')
        print(type(self.dataFrame))
        print(self.dataFrame.head())
        print(self.dataFrame.shape)

class DataSetSymbol:
    def __init__(self,symbol):
        self.symbol = symbol
        self.data = {}

d = DataPoint('AAPL')
d.pull_data()
