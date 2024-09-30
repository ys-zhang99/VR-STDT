import numpy as np
import pandas as pd
from scipy import stats

def deleteNum(num,labels):
    num = np.sort(num)
    for i in range(len(labels)):
        num = np.delete(num,np.where(preds_index==labels_index[i]))
    num = np.unique(num)
    return num


data1 = pd.Series(np.random.rand(100)*100).sort_values()
data2 = pd.Series(np.random.rand(100)*50).sort_values()
data3 = pd.Series(np.random.rand(100)*20).sort_values(ascending=False)
data = pd.DataFrame({'X':data1.values,
                     'Y':data2.values,
                     'Z':data3.values,})
data_path = './data'
air_quality_data = pd.read_csv('{}/airquality.csv'.format(data_path), nrows=278023)
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
corr = np.corrcoef(input_data)
dataframe = pd.DataFrame(corr,index=index,columns=index)
#dataframe.to_csv('corr.csv')
labels = (dataframe.mean(axis=0).sort_values(ascending=False)).iloc[:3]
labels_index = np.array(labels.index)
preds_index = []
#print(dataframe)
print(labels_index)
for i in labels_index:
    dt = dataframe.loc[i]
    dt_index = np.array(dt[(dt>0.8) & (dt<1)].sort_values(ascending=False).index)[:4]
    preds_index.extend(dt_index)

preds_index = np.array(preds_index).reshape(-1)
preds_index = np.sort(preds_index)
preds_index = deleteNum(preds_index,labels_index)
print(preds_index)


# 正态性检验
#u1,u2,u3 = data['X'].mean(),data['Y'].mean(),data['Z'].mean()  # 计算均值
#std1,std2,std3 = data['X'].std(),data['Y'].std(),data['Z'].std()  # 计算标准差

# 正态性检验 → pvalue >0.05,则数据服从正态分布
# pearson相关系数:
## data.corr(method='pearson', min_periods=1) method默认pearson
## method : {‘pearson’, ‘kendall’, ‘spearman’} 

#print("相关系数矩阵:\n",data.corr() ) # 给出相关系数矩阵

# 计算"X"与"Y"之间的相关系数
#print('\n计算"X"与"Y"之间的相关系数:',data["X"].corr(data["Y"]))

# 给出Y变量与其他变量之间的相关系数
#print("\nY变量与其他变量之间的相关系数:\n",data.corr()["Y"])