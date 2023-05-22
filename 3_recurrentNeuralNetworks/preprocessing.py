# Building a Recurrent Neural Network
import os
import numpy as np
import pandas as pd

# define constants
DATASOURCE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    ), '0_data', 'rnn_studies','0_process_time_series.csv'
)

# importing the data
data = pd.read_csv(DATASOURCE, index_col=[0], parse_dates=['Timestamp'])

# select train and test indexes
train_idx = data.loc[data['cmpg_no']==1, :].index
test_idx = data.loc[data['cmpg_no']==2, :].index

# selecting the predictors variables
x_vars = [col for col in data.columns if col.find('.PV') != -1]
x_train = data.loc[train_idx, x_vars]
x_test = data.loc[test_idx, x_vars]

# making feature engineering to have the right predicted variables

# conversion
data['conversion'] = (data['EB_s1'] - data['EB_s7']) / data['EB_s1']

# ring loss
data['c8_s1'] = data['mX_s1'] + data['oX_s1'] + data['pX_s1'] + data['EB_s1']
data['c8_s7'] = data['mX_s7'] + data['oX_s7'] + data['pX_s7'] + data['EB_s7']
data['ringLoss'] = (data['c8_s1'] - data['c8_s7']) / data['c8_s1']

# make outlier and abnormal values treatment

# conversion
data.loc[data['conversion'] < 0.2, 'conversion'] = np.nan

q1, q3 = np.quantile(data['conversion'], q=[.25, 0.75])
iqr = q3 - q1
k = 1.5
max_, min_ = q3 + (k * iqr), q1 - (k * iqr)
data.loc[((data['conversion'] < min_ ) | (data['conversion'] > max_)), 'conversion'] = np.nan

# ring Loss
q1, q3 = np.quantile(data['ringLoss'], q=[.25, 0.75])
iqr = q3 - q1
k = 1.5
max_, min_ = q3 + (k * iqr), q1 - (k * iqr)
data.loc[((data['ringLoss'] < min_ ) | (data['ringLoss'] > max_)), 'ringLoss'] = np.nan

data[['conversion', 'ringLoss']] = \
    (data[['conversion', 'ringLoss']].fillna(method='ffill') + 
     data[['conversion', 'ringLoss']].fillna(method='bfill')) / 2

# extract target variables
y_train_conversion = data.loc[train_idx, 'conversion']
y_test_conversion = data.loc[test_idx, 'conversion']
y_train_ringLoss = data.loc[train_idx, 'ringLoss']
y_test_ringLoss = data.loc[test_idx, 'ringLoss']

# save preprocessed data
x_train.to_csv(
    os.path.join(
        os.path.dirname(DATASOURCE),
        'x_train.csv'
    ),
    index=False
)
x_test.to_csv(
    os.path.join(
        os.path.dirname(DATASOURCE),
        'x_test.csv'
    ),
    index=False
)
y_train_conversion.to_csv(
    os.path.join(
        os.path.dirname(DATASOURCE),
        'y_train_conversion.csv'
    ),
    index=False
)
y_test_conversion.to_csv(
    os.path.join(
        os.path.dirname(DATASOURCE),
        'y_test_conversion.csv'
    ),
    index=False
)
y_train_ringLoss.to_csv(
    os.path.join(
        os.path.dirname(DATASOURCE),
        'y_train_ringLoss.csv'
    ),
    index=False
)
y_test_ringLoss.to_csv(
    os.path.join(
        os.path.dirname(DATASOURCE),
        'y_test_ringLoss.csv'
    ),
    index=False
)
