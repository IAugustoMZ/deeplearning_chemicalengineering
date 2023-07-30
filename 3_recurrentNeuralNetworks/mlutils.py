import os
import pickle
import numpy as np
import pandas as pd
from globals import *
import scipy.stats as sts
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential

class ML_utils:

    def __init__(self) -> None:
        """
        class with utilitary ML methods
        """
        pass

    def create_lag_structure(self,
                             data: np.array,
                             memory_steps: int,
                             future_steps: int=1) -> tuple:
        """
        create a data structure with lagged features
        (this version works only for single time series)

        Parameters
        ----------
        data : np.array
            original time series data
        memory_steps : int
            number of past value steps to consider
        future_steps : int, optional
            number of future steps to predict, by default 1

        Returns
        -------
        tuple
            lagged features
        """
        x = []
        y = []
        for i in range(memory_steps, data.shape[0]-future_steps):
            x.append(data[i-memory_steps:i, 0])
            y.append(data[i+future_steps-1, 0])

        return np.array(x), np.array(y)
    
    def build_RNN(self,
                  architecture: tuple,
                  input_shape: tuple,
                  regularization_level: float=0.2) -> None:
        """
        builds the RNN according to architecture options

        Parameters
        ----------
        architecture : tuple
            tuple with number of LSTM cells
            in each layer
        input_shape : tuple
            input shape tuple
        regularization_level : float, optional
            level of Dropout rate, by default 0.2
        """
        
        # start building our RNN
        self.regressor = Sequential()

        # add the stacked LSTMs cells and a dropout regularization
        for i in architecture:

            # add LSTMs layers according to their position
            if i == architecture[0]:
                self.regressor.add(LSTM(
                    units=i,
                    return_sequences=True,
                    input_shape=(input_shape[1], input_shape[2])
                ))
            elif i == architecture[-1]:
                self.regressor.add(LSTM(
                    units=i
                ))
            else:
                self.regressor.add(LSTM(
                    units=i,
                    return_sequences=True
                ))

            # add Dropout layer
            self.regressor.add(Dropout(regularization_level))

        # add the ouput layer
        self.regressor.add(Dense(units=1))

        # compile the model
        self.regressor.compile(optimizer='adam', loss='mean_squared_error')

    def fit_model(self,
                  x_train: np.array,
                  y_train: np.array,
                  x_val: np.array,
                  y_val: np.array,
                  N_EPOCHS : int = 50,
                  BATCH_SIZE : int = 32) -> None:
        """
        fits the RNN with the options

        Parameters
        ----------
        x_train : np.array
            train features
        y_train : np.array
            train targets
        x_val : np.array
            validation features
        y_val : np.array
            validation targets
        N_EPOCHS : int, optional
            number of training epochs, by default 50
        BATCH_SIZE : int, optional
            batch size, by default 32
        """
        
        self.history = self.regressor.fit(
            x=x_train,
            y=y_train,
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val)
        )

        # save training history
        with open(os.path.join(MODEL_SINK, 'history_{}'.format(N_EPOCHS)), 'wb') as history_file:
            pickle.dump(self.history, history_file)

        # save model weights and model architecture
        self.regressor.save(
            os.path.join(MODEL_SINK, 'modelRNN_{}.h5'.format(N_EPOCHS))
        )
        self.regressor.save_weights(
            os.path.join(MODEL_SINK, 'modelRNN_weights_{}'.format(N_EPOCHS))
        )

    def forecast(self,
                 n: int,
                 model_config: dict,
                 x: np.array) -> pd.DataFrame:
        """
        make the forecast using a trained model
        and considering a confidence interval

        Parameters
        ----------
        n : int
            number of periods in the 
            future to be forecasted
        model_config : dict
            dictionary containing the configurations
            about the model
        x : np.array
            first input for the forecast

        Returns
        -------
        pd.DataFrame
            data frame containing the results
        """

        # extract information about models
        scaler = model_config.get('scaler')
        model = model_config.get('model')
        lim_inf = model_config.get('low')
        lim_sup = model_config.get('high')
        avg = model_config.get('mean')
        alpha = model_config.get('alpha')

        # create the lists to store the results
        results_avg = []
        results_min = []
        results_max = []

        # get the necessary shape
        shape_ = x.shape

        # calculate the std. deviation
        z_value = abs(sts.t.ppf(q=alpha, df=shape_[1]))
        s_min = lim_inf * (np.sqrt(shape_[1])) / z_value
        s_max = lim_sup * (np.sqrt(shape_[1])) / z_value

        for i in range(1, n+1):

            # make the forecast
            forecast = model.predict(x)

            # get the real value
            ypred = scaler.inverse_transform(forecast)[0][0]

            # get the corrected predictions and the 
            # confidence limits
            SE_min = z_value * np.sqrt(i) * s_min / np.sqrt(shape_[1])
            SE_max = z_value * np.sqrt(i) * s_max / np.sqrt(shape_[1])
            ypred += avg
            ypred_min = ypred - SE_min
            ypred_max = ypred + SE_max

            # store the predictions for the the average
            # the min and the max
            results_avg.append(ypred)
            results_min.append(ypred_min)
            results_max.append(ypred_max)

            # update the input array
            x = list(x[0])
            x.pop(0)
            x.append(forecast[0][0])
            x = np.array(x).reshape(shape_)

        # create final dataframe of predictions
        forecasts = pd.DataFrame(results_avg, columns=['Predictions'])
        forecasts['+{}%'.format(int(100*(1-alpha)))] = results_max
        forecasts['-{}%'.format(int(100*(1-alpha)))] = results_min

        return forecasts