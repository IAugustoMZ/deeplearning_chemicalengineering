import os
import pickle
import numpy as np
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential

# define constants
DATASOURCE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    ), '0_data', 'rnn_studies','BrentOilPrices.csv'
)
MODEL_SINK = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    ), '1_models', 'rnn_studies'
)

# define utils class
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
        with open(os.path.join(MODEL_SINK, 'history'), 'wb') as history_file:
            pickle.dump(self.history, history_file)

        # save model weights and model architecture
        self.regressor.save(
            os.path.join(MODEL_SINK, 'modelRNN.h5')
        )
        self.regressor.save_weights(
            os.path.join(MODEL_SINK, 'modelRNN_weights')
        )