# Stock Market Price Prediction

A college project that uses data acquired from the Yahoo Finance API to predict 
the closing price of a stock in the National Stock Exchange, India. The main aim 
of the project is to compare the performance of traditional forecasting 
techniques for stock pricing with advanced AI based models,and determine their 
viability in real world scenarios.

<h2> Some Screenshots:</h2>


![Screenshot 2023-08-09 at 12 50 10 AM](https://github.com/chcheetah/Stock-Market-Price-Prediction/assets/79366050/3963abe1-ffe3-42b5-870e-07e9ba911270)

Prediction performance chart comparing model output with real world data.

|Type|Color|
|----|-----|
|Model Prediction|🟧|
|Real World Data|🟦|

<h2>Results</h2>

|Paper |Algorithm| RMSE| MAE|
|------|---------|-----|----|
|Proposed Model |CNN| 0.0838| 0.0534
||CNNLSTM| 0.0891| 0.0575|
||LSTM |0.0827| 0.0557|
||SVR |0.0940 |0.0725|
|Banik et al, 2022*| LR| 0.0753| 0.0542|
||MA |1.4651| 1.2103|
||XGBR |1.0362| 0.8744|
||SVR |0.6015| 0.5287|
||ARIMA| 0.5741| 0.4647|
||ETS| 0.5754| 0.4631|
||Meanf| 0.5754| 0.4631|
||BoxCox| 0.5493| 0.4368|
||LSTM| 0.0413| 0.0324|



<h2> Condensed Project Status: <font > Online </font>, working parttime</h2>


|Goal|Achieved?|Further Improvements|
|----|---------|------|
|Predict the closing price of a stock| ✅|Better Model Optimization|
|Use realtime stock market data| ✅|Use more precise data, maybe different markets as well?|
|User selectable Models : LSTM, LSTM-CNN, CNN, SVM| ✅| Add more models to the program,like Bi-LSTM| 
|Provides future predictions daily| ✅| Adding more time periods: weekly, monthly, yearly|
|Has a user-friendly GUI| ✅| Improving upon GUI|
|Able to view indicators for a given stock| 🚧| Building a better layout for the graphs generated|

<h2> Developer: </h2>

| Name| Github id|
|---------------|-----------|
|Harshiv Chandra| @chcheetah|

<h2> References: </h2>

<pre> * Banik, Shouvik, Nonita Sharma, Monika Mangla,
Sachi Nandan Mohanty, and S. Shitharth. "LSTM
based decision support system for swing trading in
stock market." (“LSTM based decision support system 
for swing trading in stock market ...”) Knowledge-
Based Systems 239 (2022): 107994.</pre>
