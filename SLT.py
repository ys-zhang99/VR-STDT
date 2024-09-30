from sklearn import preprocessing
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean



data_path = './data'
scaler = preprocessing.MinMaxScaler(feature_range=(0,200))
air_quality_data = pd.read_csv('{}/airquality.csv'.format(data_path), nrows=278023)

# remove data from 1022 that with lot of null data
air_quality_data = air_quality_data[air_quality_data['station_id'] != 1022]
columns = ['PM25_Concentration', 'PM10_Concentration',
            'NO2_Concentration', 'CO_Concentration',
            'O3_Concentration', 'SO2_Concentration']
# pivot the data
pivot_air_data = air_quality_data.pivot(index='time', columns='station_id', values=columns)
# linear interpolate to fill the loss value
pivot_air_data1 = pivot_air_data.interpolate(method='linear').dropna()

air_quality_data = pivot_air_data1.stack(level=1).reset_index().sort_values(by=['station_id', 'time'])
input_data = []
index = []
#print(np.array(air_quality_data[(air_quality_data['station_id'] == 1001)].loc[:,'PM25_Concentration'].values).shape)
for i in range(1001,1036):
    #input_data.append(np.array(air_quality_data[(air_quality_data.station_id == i)].drop(['station_id', 'time'], axis=1).values))
    if i!=1022:
        index.append(i)
        input_data.append(np.array(air_quality_data[(air_quality_data.station_id == i)].loc[:,'PM25_Concentration'].values))
input_data = np.array(input_data)
print(input_data.shape)
corr = np.corrcoef(input_data)
#print(np.corrcoef(input_data))
dataframe = pd.DataFrame(corr,index=index,columns=index)
#dataframe.to_csv('corr.csv')
labels = (dataframe.mean(axis=0).sort_values(ascending=False)).iloc[:3]
labels_index = np.array(labels.index)
preds_index = []
#print(dataframe[>0.8])
print(labels_index)
for i in labels_index:
    dt = dataframe.loc[i]
    dt_index = np.array(dt[(dt>0.8) & (dt<1)].sort_values(ascending=False).index)[:4]
    preds_index.extend(dt_index)

preds_index = np.array(preds_index).reshape(-1)
preds_index = np.sort(preds_index)
for i in range(len(labels_index)):
    preds_index = np.delete(preds_index,np.where(preds_index==labels_index[i]))
preds_index = np.unique(preds_index)
print(preds_index)

input_data = []
label_data = []
for i in preds_index:
#for i in range(1018,1036):
    #input_data.append(np.array(air_quality_data[(air_quality_data.station_id == i)].drop(['station_id', 'time'], axis=1).values))
    if i !=1022:
        input_data.append(scaler.fit_transform(np.array(air_quality_data[(air_quality_data.station_id == i)].drop(['station_id', 'time'], axis=1).values)))
for i in labels_index:
    if i !=1022:
        label_data.append(scaler.fit_transform(np.array(air_quality_data[(air_quality_data.station_id == i)].drop(['station_id', 'time','PM10_Concentration','NO2_Concentration', 'CO_Concentration',
            'O3_Concentration', 'SO2_Concentration'], axis=1).values)))

""" for i in range(1018,1036):
    #input_data.append(np.array(air_quality_data[(air_quality_data.station_id == i)].drop(['station_id', 'time'], axis=1).values))
    if i !=1022:
        input_data.append(scaler.fit_transform(np.array(air_quality_data[(air_quality_data.station_id == i)].drop(['station_id', 'time'], axis=1).values)))
for i in range(1001,1018):
    if i !=1022:
        label_data.append(scaler.fit_transform(np.array(air_quality_data[(air_quality_data.station_id == i)].drop(['station_id', 'time','PM10_Concentration','NO2_Concentration', 'CO_Concentration',
            'O3_Concentration', 'SO2_Concentration'], axis=1).values))) """

input_data = np.array(input_data)
label_data = np.array(label_data) 
""" scl2 =  preprocessing.StandardScaler(with_mean=np.mean(input_data[0]),with_std=np.std(input_data[0]))
print(input_data[0])
print(scl2.inverse_transform(input_data[0])) """
input_data = input_data.transpose(1,0,2)
label_data = label_data.transpose(1,0,2)
enstep = 6
destep = 1
steps = 1
x = []
y = []
tx = []
ty = []
split_rr = int(input_data.shape[0]//10)
train_input = input_data[:10*split_rr]
train_label = label_data[:10*split_rr]
time_train = train_input.shape[0]
test_input = input_data[9*split_rr:]
test_label = label_data[9*split_rr:]
time_test = test_input.shape[0]

y = test_label.transpose(1,0,2)
""" decomp = series_decomp(15)
seasonal_init, trend_init = decomp(torch.from_numpy(y))
seasonal_init = np.array(seasonal_init)
trend_init = np.array(trend_init)

show_origin = y[0,:,0].reshape(845)
print(show_origin.shape)
show_season = seasonal_init[0,:,0].reshape(y.shape[1])
show_trend = trend_init[0,:,0].reshape(y.shape[1])

print(seasonal_init.shape[1])

x_axis = range(0,seasonal_init.shape[1])
origin = plt.figure('oringin').subplots()
season = plt.figure('season').subplots()
trend = plt.figure('trend').subplots()
origin.plot(x_axis,show_origin)
season.plot(x_axis,show_season)
trend.plot(x_axis,show_trend)
plt.show() """
#from statsmodels.datasets import co2
#import matplotlib.pyplot as plt
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()
#data = co2.load().data
data = y[0,:].squeeze()
#data = data.resample('M').mean().ffill()

from statsmodels.tsa.seasonal import STL
res = STL(data,period=7,robust=False).fit()
res.plot()
seasonal = res.seasonal
trend = res.trend
resid = res.resid

data_1 = resid+seasonal
print(data_1.shape)
x_axis = range(0,data.shape[0])
origin = plt.figure('oringin').subplots()
quzao  = plt.figure('quzao').subplots()
origin.plot(x_axis,seasonal)
quzao.plot(x_axis,data_1)
plt.show() 