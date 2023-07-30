import os
import sys
import numpy as np
import pandas as pd
from globals import *
from  mlutils import ML_utils

# load the datasets
datasets = dict()
for data in ['train', 'test', 'validation']:
    datasets[data] = pd.read_csv(
        os.path.join(
            os.path.dirname(DATASOURCE), '{}.csv'.format(data)
        ),
        index_col=[0]
    ).values

# create a data structure with 90 past values and 1 outputs
MEMORY_STEPS = 30           # behavior of last 3 months
PREDICTION_WINDOW = 1       # predict next day
mlu = ML_utils()

x_y_datasets = dict()
for data in ['train', 'test', 'validation']:
    x, y = mlu.create_lag_structure(
        data=datasets[data],
        memory_steps=MEMORY_STEPS,
        future_steps=PREDICTION_WINDOW
    )

    # reshaping the input array (only for multiple inputs)
    x = np.reshape(x, (x.shape[0], MEMORY_STEPS, 1))

    x_y_datasets['x_{}'.format(data)] = x
    x_y_datasets['y_{}'.format(data)] = y

# build and compile the RNN
mlu.build_RNN(
    architecture=(100, 100, 50, 25),
    input_shape=x_y_datasets['x_train'].shape,
    regularization_level=0.1
)

mlu.fit_model(
    x_train=x_y_datasets['x_train'],
    y_train=x_y_datasets['y_train'],
    x_val=x_y_datasets['x_validation'],
    y_val=x_y_datasets['y_validation'],
    N_EPOCHS=75,
    BATCH_SIZE=8
)

print('OK')