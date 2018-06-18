import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as smf
from scipy import stats
from sklearn import linear_model
from sklearn import datasets, linear_model


dataset1=pd.read_csv('MT_cleaned.csv')
dataset2=pd.read_csv('VT_cleaned.csv')
size1=len(dataset1)
size2=len(dataset2)

#number of male drives involve in traffic stop
size1_1=(dataset1.groupby('driver_gender').size())
print(size1_1.M/size1)

#arrests in Montana during a traffic having  out of state plates
size1_2=(dataset1.groupby('out_of_state').size())
print(size1_2[True]/size1)

# Chi-square 
arrest1=(dataset1.groupby('is_arrested').size())
arrest2=(dataset2.groupby('is_arrested').size())
a=arrest1[True]/size1
b=arrest2[True]/size2

print(chisquare([a, b]))

# viloations including speeding 
size1_3=(dataset1.groupby('violation').size())
print(size1_3.Speeding/size1)

#number of stops of DUI
size2_3=(dataset2.groupby('violation').size())
print((size1_3.DUI/size1)/(size2_3.DUI/size2))

## extrapolate

# data imported form manufacture year of vehicles involved in traffic stops 
dataset1_1=pd.read_csv('manufacture.csv')
Y=dataset1_1[:,1].reshape(-1, 1)
X=dataset1_1[:,0].reshape(-1, 1)
X_train=x[:8]
Y_train=y[:8]
Y_test=y[8:12]
X_test=x[8:12]

lm1=sm.OLS(Y_train,X_train ).fit()
lm1.params
lm2 = LinearRegression()
lm2.fit(X_train, Y_train)

print(lm2.intercept_)
print(lm2.coef_)
lm1.pvalues
lm2.predict(X_test)


# differnce of maximum and minumum

dataset1['Hour'], dataset1['Mins'] = dataset1['stop_time'].str.split(':', 1).str

dataset2['Hour'], dataset2['Mins'] = dataset2['stop_time'].str.split(':', 1).str

datset_combine=pd.concat([dataset1,dataset2])


f = datset_combine['Hour'].value_counts()

Max=max(f)
Min=min(f)
print(Max-Min)

## county area in kilometers
size1_5=dataset1.groupby('county_name').std()
del size1_5['county_fips']
del size1_5['police_department']
del size1_5['driver_age_raw']
del size1_5['driver_age']



