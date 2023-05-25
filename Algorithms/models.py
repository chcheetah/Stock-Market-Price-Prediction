import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Flatten, Dense, Activation
from keras.layers import CuDNNLSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.model_selection import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from sklearn.svm import SVR
import tensorflow as tf
import os
import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error



class modelling:
    '''
    This class defines the functions needed to create a model forecasting
    the next day close predicted for a given stock ticker.
    '''
    def dataload(self,time_steps=10):
        '''
        Defines the data preprocessing algorithms
        needed for the model.
        '''
        data = pd.read_csv(os.getcwd()+'\\Algorithms\\train.csv', index_col = ["Unnamed: 0"])
        data= data.dropna(axis=0)
        data= data.values[:, 0:]
        y= data[:, 3]
        print(y.shape)
        X= data[:,:]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        maxi = np.max(y,axis = 0)
        mini = np.min(y,axis = 0)
        y = (y- np.min(y,axis=0))/(np.max(y, axis=0)-np.min(y,axis=0))
        X_new= np.zeros((X.shape[0] - time_steps +1, time_steps, X.shape[1]))
        y_new= np.zeros((y.shape[0] -time_steps +1,))
        for ix in range(X_new.shape[0]):
            for jx in range(time_steps):
                X_new[ix, jx, :]= X[ix +jx, :]
            y_new[ix-1]= y[ix + time_steps -1]
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2)
        return X_new, y_new, time_steps, X_train, X_test, y_train, y_test, maxi,mini, X, y
    def model(self,architecture = "LSTM", option = "GENERATE", time_steps = 10,feature_count = 6):
        '''
        Defines and loads model into memory as required.
        Currently supports following architectures,
            LSTM
            CNN + LSTM
            CNN
            SVM
        Parameters :
        @param architecture = (LSTM|CNN+LSTM|CNN|SVM)
        @param option = (LOAD|GENERATE)
        @param time_step = no. of time steps taken into consideration on input data (AKA window)
        '''
        path = os.getcwd()+"\\savedmodels"
        if(option == "GENERATE"):
            model = Sequential()
            if(architecture == "LSTM"):
                print()
                model.add(LSTM(100, input_shape= (time_steps,feature_count), return_sequences=True))
                model.add(LSTM(50, return_sequences=False))    
                model.add(Dense(1))
                model.add(Activation('linear'))
                model.summary()
                model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
                return model
            elif(architecture == "CNNLSTM"):
                model.add(Conv1D(64, 3, padding='same', input_shape=(time_steps, feature_count)))
                model.add(MaxPooling1D(pool_size=2))
                model.add(LSTM(100, return_sequences=True))
                model.add(Conv1D(32, 3, padding='same'))    
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())    
                model.add(Dense(1))
                model.add(Activation('linear'))
                model.summary()
                model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
                return model
            elif(architecture == "CNN"):
                model.add(Conv1D(64, 3, padding='same', input_shape=(time_steps, feature_count)))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Conv1D(32, 3, padding='same'))    
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())    
                model.add(Dense(1))
                model.add(Activation('linear'))
                model.summary()
                model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
                return model
            elif(architecture == "SVM"):
                model = SVR(kernel = 'rbf')
                return model
            else:
                raise AssertionError("Architecture selected does not exist. \n You must select either LSTM, CNNLSTM, CNN or SVM")
        else:
            if(algo == "SVM"):
                filename = path+"\\"+algo+"\\finalized_model.sav"
                model = joblib.load(filename)
                return model
            model = tf.keras.models.load_model(path+"\\"+architecture)
            model.summary()
            return model
    def new_preds(self,data, maximum,minimum,model,time_steps=10,features=6):
        last_sequence =np.asarray( [data[-i].tolist() for i in range(1,time_steps+1)])# Take the last sequence from the reshaped data of window 
        input_data = last_sequence
        input_data = input_data.reshape((1, time_steps, features))
        predicted_output = model.predict(input_data)
        predicted_output = (predicted_output)*(maximum-minimum) + minimum
        print("Predicted close for the next day:", predicted_output)
        return predicted_output
    def new_preds_SVR(self,data,maximum,minimum,model,features=6):
        last_sequence = data[-1]
        print(last_sequence)
        input_data = last_sequence
        input_data = input_data.reshape((1,features))
        predicted_output = model.predict(input_data)
        predicted_output = (predicted_output)*(maximum-minimum) + minimum
        print("Predicted close for the next day:", predicted_output)
        return predicted_output
    def __init__(self,algo):
        path = os.getcwd()+"\\savedmodels"
        X_new, y_new, n_lookback, X_train, X_test, y_train, y_test, maxi,mini, X, y = self.dataload()
        Model = self.model(algo,feature_count=X.shape[1])
        if(algo=="SVM"):
            xtr, xt, ytr, yt = train_test_split(X_new, y_new, test_size=0.2)
            Model.fit(xtr[:,0],ytr)
            filename = path+"\\"+algo+"\\finalized_model.sav"
            joblib.dump(Model, filename)
            y_pred= Model.predict(X_new[:,0])
            mse = mean_squared_error(y_pred,y_new,squared=True)
            rmse = mean_squared_error(y_pred,y_new,squared=False)
            mae = mean_absolute_error(y_pred,y_new)
            print("MSE:",mse,"RMSE:",rmse,"MAE:",mae)
            plt.plot((y_pred)*(maxi-mini) + mini, 'r-')
            plt.plot((y_new)*(maxi-mini) + mini, 'b-')
            plt.ylabel('Stock Price')
            plt.xlabel('Days')
            plt.title("Prediction vs actual market data for"+algo)
            plt.show()
            plt.plot ((y_pred)*(maxi-mini) + mini,(y_new)*(maxi-mini) + mini,'bo')
            plt.xlabel('y_pred')
            plt.ylabel('y_original')
            plt.title("Prediction vs actual market data for"+algo)
            plt.show()
            self.preds = self.new_preds_SVR(X_new[:,0],maxi,mini,Model)
            
        else:
            Model.fit(X_train,y_train,epochs = 35)
            Model.save(path+"\\"+algo)
            score= Model.evaluate(X_test, y_test)
            print("\n\nScore\n\n")
            print(score)
            y_pred= Model.predict(X_new)
            plt.plot((y_pred)*(maxi-mini) + mini, 'r-')
            plt.plot((y_new)*(maxi-mini) + mini, 'b-')
            plt.ylabel('Stock Price')
            plt.xlabel('Days')
            plt.title("Prediction vs actual market data for"+algo)
            plt.show()
            plt.plot ((y_pred)*(maxi-mini) + mini,(y_new)*(maxi-mini) + mini,'bo')
            plt.xlabel('y_pred')
            plt.ylabel('y_original')
            plt.title("Prediction vs actual market data for"+algo)
            plt.show()
            self.preds = self.new_preds(X,maxi,mini,Model)
            
