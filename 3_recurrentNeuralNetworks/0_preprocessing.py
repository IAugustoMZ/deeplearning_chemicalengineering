# Preprocessing the data for RNN modeling
import joblib
import pandas as pd
from globals import *
from sklearn.preprocessing import MinMaxScaler

# load the data set and split
data = pd.read_csv(DATASOURCE, parse_dates=['Date'])

# split the data in very impactful periods
begin_pandemic = pd.to_datetime('01-10-2020') # WHO official begin of COVID pandemic
end_oil_shock = pd.to_datetime('01-01-2009') # oil market shock

train = data.loc[data['Date'] < begin_pandemic, :]
test = data.loc[data['Date'] >= begin_pandemic, ['Price']]
validation = train.loc[train['Date'] >= end_oil_shock, ['Price']]
train = train.loc[train['Date'] < end_oil_shock, ['Price']]

# apply normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train)

train_norm = pd.DataFrame(scaler.transform(train), columns=['Price'])
validation_norm = pd.DataFrame(scaler.transform(validation), columns=['Price'])
test_norm = pd.DataFrame(scaler.transform(test), columns=['Price'])

# save data
train_norm.to_csv(os.path.join(
    os.path.dirname(DATASOURCE), 'train.csv'
), index=True)
validation_norm.to_csv(os.path.join(
    os.path.dirname(DATASOURCE), 'validation.csv'
), index=True)
test_norm.to_csv(os.path.join(
    os.path.dirname(DATASOURCE), 'test.csv'
), index=True)

# save scaler model
joblib.dump(scaler, os.path.join(MODEL_SINK, 'scaler.m'))