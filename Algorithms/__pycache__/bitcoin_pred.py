import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import requests as req
from prettytable import PrettyTable
import datetime as dt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from tkinter import *
from math import *
def bitrate_extract(typ="BTC"):
    headers={'User-Agent':'Mozilla/5.0(Macintosh;Intel Mac OS X 10_11_2) AppleWebKit/601.3.9(KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'}

    if(typ == "BTC"):
        url='https://finance.yahoo.com/quote/BTC-USD/history/'
    elif(typ == "ETH"): #ethereum
        url='https://finance.yahoo.com/quote/ETH-USD/history/'
    elif(typ == "ADA"): # cardano
        url='https://finance.yahoo.com/quote/ADA-USD/history/'
    elif(typ== "XRP"): # XRP
        url='https://finance.yahoo.com/quote/XRP-USD/history/'
    elif(typ == "LTC"): # Litecoin
        url='https://finance.yahoo.com/quote/LTC-USD/history/'

        
    resp=req.get(url,headers=headers)
    soup=BeautifulSoup(resp.content,"html.parser")
    find=soup.find("table",attrs={"class":"W(100%) M(0)"})
    find_data=find.tbody.find_all("tr")
    data=[]
    i=0
    j=0
    table=PrettyTable(["Date","Open","High","Low","Close","Adj Close","Volume"])
    while j<100:
        for td in find_data[j].find_all("td"):
            data.append(td.text.replace('\n',' '))
        j=j+1
        table.add_row(data)
        data=[]
        i=0
    print(table)
    return table

def bit_pred(table):
    dtr=[]
    dtr2=[]
    dtr3=[]
    for i in range(100):
        a=table.rows[i][4]
        b=table.rows[i][1]
        dtr.append(((float(a.replace(',','')))+(float(b.replace(',',''))))/2)
    day=list(range(1,101))
    plt.plot(day,dtr)
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title("Bitcoin price predictor")
    slope,intercept=np.polyfit(day,dtr,1)
    plt.show()
def writi(table):
    avg = []

    f = open('data.csv',mode='w+',newline='')
    w = csv.writer(f)
    for i in range(100):
        datx = []
        a=table.rows[i][4]
        b=table.rows[i][1]
        datx.append(i+1)
        datx.append(((float(a.replace(',','')))+(float(b.replace(',',''))))/2)
        w.writerow(datx)

def regression(table,typ,r=101):
    avg = []
    date = []
    data = pd.read_csv('data.csv')# load data set
    X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)
    r_arr = []
    r_arr.append(r)
    ssa = np.array(r_arr)
    Y_pred = linear_regressor.predict(X)
    y_userpred = linear_regressor.predict(ssa.reshape(-1,1))
    dt = []
    dt.append(r)
    predval(r,round(y_userpred[0][0],2),typ)

def predval(r,y,typ):
    if(typ == "BTC"):
        d = 'Bitcoin'
    elif(typ == "ETH"): #ethereum
        d = 'Ethereum'
    elif(typ == "ADA"): # cardano
        d = 'Cardano'
    elif(typ== "XRP"): # XRP
        d = 'XRP'
    elif(typ == "LTC"): # Litecoin
        d = 'Litecoin'
    l = Tk()
    l.title("Predicted Value")
    l.resizable(0,0)
    l.geometry("500x200")
    k = Label(l,text="At day "+str(r)+ " "+d+" shall be worth "+str(y)+" USD ",bg='black',fg='yellow')
    k.config(font=("Courier", 12))
    k.pack(expand=True,fill=BOTH)
    l.deiconify()
    l.mainloop()
def graph(table,typ="BTC"):
    day=[]
    op=[]
    hi=[]
    lo=[]
    cl=[]
    dtr=[]
    if(typ == "BTC"):
        d = 'Bitcoin'
    elif(typ == "ETH"): #ethereum
        d = 'Ethereum'
    elif(typ == "ADA"): # cardano
        d = 'Cardano'
    elif(typ== "XRP"): # XRP
        d = 'XRP'
    elif(typ == "LTC"): # Litecoin
        d = 'Litecoin'
    for i in range(99,-1,-1):
        a=table.rows[i][4]
        b=table.rows[i][1]
        dtr.append(((float(a.replace(',','')))+(float(b.replace(',',''))))/2)
        day.append(table.rows[i][0].replace(',',''))
        op.append(float(table.rows[i][1].replace(',','')))
        hi.append(float(table.rows[i][2].replace(',','')))
        lo.append(float(table.rows[i][3].replace(',','')))
        cl.append(float(table.rows[i][4].replace(',','')))
    print(op)
    sr=pd.Series(dtr)
    fig=go.Figure(
        data=[
            go.Candlestick(
                x=day,
                open=op,
                high=hi,
                low=lo,
                close=cl
                ),
            go.Scatter(
                x=day,
                y=sr.rolling(window=10).mean(),
                mode='lines',
                name='10 day Simple Moving Average',
                line={'color':'#ff006a'}
                ),
            go.Scatter(
                x=day,
                y=sr.rolling(window=20).mean(),
                mode='lines',
                name='20 day Simple Moving Average',
                line={'color':'#1900ff'}
                )
            ]
        )
    fig.update_layout(
        title='Graph for ' + d,
        xaxis_title='Date',
        yaxis_title='Price (in USD)',
        xaxis_rangeslider_visible=False
        )
    fig.update_yaxes(tickprefix='$')
    fig.show()
    

    
    





