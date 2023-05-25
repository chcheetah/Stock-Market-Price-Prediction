from tkinter import *
from tkinter import ttk
from Algorithms.models import *
from Algorithms.yahoofetch import *
from Algorithms.bitcoin_pred import *
import time
def main():
    a = Tk()
    a.resizable(0,0)
    a.config(bg = 'black')
    a.title("Stock Predictor")
    a.geometry("500x300")
    k = Label(a,text="Choose your stock ticker \n from the drop down list",fg="yellow",bg='black')
    k.config(font=("Courier", 12))
    m = []
    for i in pd.read_csv("tickers.csv")["Ticker"]:
        m.append(i)
    combo = ttk.Combobox(
    state="readonly",
    values=m,
    )
    kb = Label(a,text="Choose your algorithm from the drop down list",fg="yellow",bg='black')
    kb.config(font=("Courier", 12))
    combot = ttk.Combobox(
    state="readonly",
    values=["LSTM", "CNNLSTM", "CNN", "SVM"],
    )
    statl = StringVar()
    statl.set("Status : Ready")
    L5 = Button(a,text="Next",bg = 'green',fg='black',command=lambda: ask(combo.get(),combot.get(),a,statl))
    abt = Button(a,text="About",bg = 'purple',fg='black',command=lambda: about())
    k.pack(side = "top",padx=5,pady=40)
   # L1.pack(padx=2,pady=2)
   # L2.pack(padx=2,pady=2)
    #L3.pack(padx=2,pady=2)
    #L4.pack(padx=2,pady=2)
    combo.pack()
    kb.pack()
    combot.pack()
    L5.pack(padx=2,pady=2)
    abt.pack(padx=2,pady=10)
    statuslabel = Label(a,text="Status : Ready")
    statuslabel.config(font=("Courier", 9))
    statuslabel.pack(padx=2,pady=14)
    a.mainloop()
    
def ask(typ,algo,dsw,stat):
    print(typ)
    print(algo)
    time.sleep(2)
    stat.set("Status : Fetching data from API...")
    dsw.update()
    mm = ticker_list().get_data_and_Store(typ)
    time.sleep(2)
    stat.set("Status : Loading data into table...")
    dsw.update()
    c = bitrate_extract()
    time.sleep(2)
    stat.set("Status : Creating graph with indicators...")
    dsw.update()
    graph(c,mm[0],typ)
    time.sleep(2)
    stat.set("Status : Predicting closing price with "+algo+" based model...")
    dsw.update()
    m = modelling(algo)
    time.sleep(2)
    stat.set("Status : Completed Predictions...")
    dsw.update()
    rew = Tk()
    rew.title("Result")
    rew.resizable(0,0)
    rew.config(bg='black')
    rew.geometry("300x100")
    rw = Label(rew,text="Predicted close price for \n "+typ+", using "+algo+"\n algorithm is ₹ "+str(m.preds[0][0]),bg="black",fg="yellow")
    rw.config(font=("Courier", 12))
    rw.pack(side='top')
  #  e = Entry(rew,bg = "black",fg ="yellow") # no need for this thus unpacked
    
    d = Button(rew,text = "OK", command=lambda: coin(rew,dsw,stat),bg = "green",fg="black")
   # e.pack()
    d.pack(side='bottom')
    rew.mainloop()
    
def about():
    abou = Tk()
    abou.title("About")
    abou.resizable(0,0)
    abou.geometry("350x350")
    iuw = Label(abou,text="Made by \n Harshiv Chandra \n Muhib Ahmed \n for Prof. Pramod Gaur \n as a part of \n a Design Project  ",bg="black",fg="grey")
    iuw.config(font=("Courier", 20))
    iuw.pack(expand=True,fill=BOTH)
    abou.mainloop()

def coin(rew,dsw,stat):
    rew.destroy()
    time.sleep(2)
    stat.set("Status : Ready")
    dsw.update_idletasks()
    dsw.deiconify()


