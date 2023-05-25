import datetime as t
import requests
import pandas as pd
import ftplib
import io
import re
import json
import datetime
import os
try:
    from requests_html import HTMLSession
except Exception:
    print("""Warning - Certain functionality 
             requires requests_html, which is not installed.
             
             Install using: 
             pip install requests_html
             
             After installation, you may have to restart your Python session.""")
class ticker_list:
    ''' Ticker List Generator, copies ticker data from Wikipedia, and
        verifies data availability for the Yahoo API.

        Run @proc refresh() to generate a new .csv file in case data does not exist.
    '''
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
    def build_url(self,ticker, start_date = None, end_date = None, interval = "1d"):
        if end_date is None:  
            end_seconds = int(pd.Timestamp("now").timestamp())
        else:
            end_seconds = int(pd.Timestamp(end_date).timestamp())
        if start_date is None:
            start_seconds = 7223400    
        else:
            start_seconds = int(pd.Timestamp(start_date).timestamp())
        site = self.base_url + ticker
        params = {"period1": start_seconds, "period2": end_seconds,
                  "interval": interval.lower(), "events": "div,splits"}
        return site, params

    def tickers_nse(self,include_company_data = False):
        '''Downloads list of tickers currently listed in NSE, from Wikipedia. '''
        nse = pd.read_html("https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_National_Stock_Exchange_of_India")[1:]
        return nse

    def get_data(self,ticker, start_date = None, end_date = None, index_as_date = True,
                 interval = "1d", headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'} ):
        '''Downloads historical stock price data into a pandas data frame.  Interval
           must be "1d", "1wk", "1mo", or "1m" for daily, weekly, monthly, or minute data.
           Intraday minute data is limited to 7 days.
           @param: ticker
           @param: start_date = None
           @param: end_date = None
           @param: index_as_date = True
           @param: interval = "1d"
        '''
        if interval not in ("1d", "1wk", "1mo", "1m"):
            raise AssertionError("interval must be of of '1d', '1wk', '1mo', or '1m'")
        # build and connect to URL
        site, params = self.build_url(ticker, start_date, end_date, interval)
        resp = requests.get(site, params = params, headers = headers)
        if not resp.ok:
            raise AssertionError(resp.json())
        # get JSON response
        data = resp.json()
        # get open / high / low / close data
        frame = pd.DataFrame(data["chart"]["result"][0]["indicators"]["quote"][0])
        # get the date info
        temp_time = data["chart"]["result"][0]["timestamp"]
        if interval != "1m":
            # add in adjclose
            frame["adjclose"] = data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]   
            frame.index = pd.to_datetime(temp_time, unit = "s")
            frame.index = frame.index.map(lambda dt: dt.floor("d"))
            frame = frame[["open", "high", "low", "close", "adjclose", "volume"]]    
        else:
            frame.index = pd.to_datetime(temp_time, unit = "s")
            frame = frame[["open", "high", "low", "close", "volume"]]
        frame['ticker'] = ticker.upper()
        if not index_as_date:  
            frame = frame.reset_index()
            frame.rename(columns = {"index": "date"}, inplace = True) 
        return frame

    def refresh(self):
        start_date = "/".join([(t.datetime.now().strftime("%d/%m/%Y")).split(sep="/")[0],(t.datetime.now().strftime("%d/%m/%Y")).split(sep="/")[1],str(int((t.datetime.now().strftime("%d/%m/%Y")).split(sep="/")[2])-1)])
        end_date = t.datetime.now()
        ff = self.tickers_nse()[0:27]
        Symbol = []
        print(len(Symbol))
        for i in ff:
            m = [(j).split("NSE:\xa0")[1] for j in list(pd.DataFrame(i)["Symbol"])]
            Symbol = Symbol + m
        print(len(Symbol))
        for tiec in Symbol:
            try:
              wwww= self.get_data(tiec+".NS",start_date=start_date,end_date=end_date)
            except: # in case it is not able to get data...
                Symbol.remove(tiec)
                print(tiec+".NS is removed")
                print(len(Symbol))
        print("Final count : ")
        print(len(Symbol))

        ee = pd.DataFrame(Symbol,columns=["Ticker"])
        
        if os.path.exists(os.getcwd()+"\\list\\"):
            ee.to_csv(os.getcwd()+"\\list\\tickers.csv")
        else:
            os.makedirs(os.getcwd()+"\\list\\")
        ee.to_csv(os.getcwd()+"\\list\\tickers.csv")
    def get_data_and_Store(self,ticker):
        start_date = "/".join([(t.datetime.now().strftime("%d/%m/%Y")).split(sep="/")[0],(t.datetime.now().strftime("%d/%m/%Y")).split(sep="/")[1],str(int((t.datetime.now().strftime("%d/%m/%Y")).split(sep="/")[2])-1)])
        end_date = t.datetime.now()
        www = self.get_data(ticker+".NS",start_date=start_date,end_date=end_date)
        www = www[www.columns[0:6]]
        www.to_csv(os.getcwd()+"\\Algorithms\\train.csv")
        return www.shape
