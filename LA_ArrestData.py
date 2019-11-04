#Name: LAPD Data Analysis
#Prepared by: Saptarshi Datta
#Date: Nov 3, 2019


import pandas as pd
import numpy as np
from scipy.stats import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

excel_file_path = (r'C:\Users\joy\Documents\Python Scripts\DataIncubator\Arrest_Data_from_2010_to_Present_sample_large.csv')

df = pd.read_csv(excel_file_path,sep=',')
df['Arrest Date'] = pd.to_datetime(df['Arrest Date'])

start_date = '2018-01-01'
end_date = '2018-12-31'

arrest_date_mask = (df['Arrest Date'] > start_date) & (df['Arrest Date'] <= end_date)
df_2018 = df.loc[arrest_date_mask]
print("The number of bookings of arrestees in 2018 were", len(df_2018.index) )

x = df_2018['Area ID'].value_counts()
print("The Area ID with the most arrests in 2018 is", x.idxmax(), "where",  x.loc[x.idxmax()], "arrests were made.")

'''
#Robbery = 3
#Burglary = 5
#Vehicle theft = 7
#Stolen property = 11

categories = [3,5,7,11]
df_2018_cg = (df_2018[df_2018['Charge Group Code'].isin(categories)])
print("The 95 quantile age for charge groups 3, 5, 7 and 11 is", df_2018_cg['Age'].quantile(0.95), "years")

#=================================================================================================

#Predelinquency 27
#Non criminal detention 26

df_2018.dropna(subset = ['Charge Group Code'],axis=0,  how='any', inplace=True)
df_2018 = df_2018.drop(df_2018[df_2018['Charge Group Code']==27].index)
df_2018 = df_2018.drop(df_2018[df_2018['Charge Group Code']==26].index)
avg_age = df_2018.groupby('Charge Group Code')['Age'].mean()
z_score = stats.zscore(avg_age)

print ('The maximum z-score is', np.amax(np.absolute(z_score)))
     


#=================================================================================================

felony = ['VIOLATION OF PAROLE:FELONY']
df_felony = (df[df['Charge Description'].isin(felony)])
df_felony = df_felony.groupby([df_felony['Arrest Date'].dt.year.rename('year')]).size().reset_index(name='counts')
    
X = df_felony['year']  
y = df_felony['counts']      

model = LinearRegression()
model.fit(X, y)

X_predict = [2019]
y_predict = model.predict(X_predict)

print('The projected number of felony arrests in', X_predict, 'is', y_predict)
'''
#=================================================================================================

bbb_lat = 34.050536
bbb_lon = -118.247861

df_2018['latitude'] = df_2018['Location'].str[1:7]
df_2018['longitude'] = df_2018['Location'].str[9:17]

delta_sigma = abs(bbb_lat-df_2018['latitude'])*3.14/180
delta_sigma_m = np.cos(abs(bbb_lon-df_2018['longitde'])*3.14/180)
delta_tau = ((bbb_lat+df_2018['latitude'])/2)

df_2018['distance'] = 6371*np.sqrt((abs(bbb_lat-df_2018['latitude'])*3.14/180)^2+(np.cos(abs(bbb_lon-df_2018['longitde'])*3.14/180)*((bbb_lat+df_2018['latitude'])/2))^2)

distance_mask = (df_2018['distance'] <= 2)
df_2018 = df.loc[distance_mask]
print("The number of arrests within 2km of the Brookings building in 2018 was", len(df_2018.index) )

#=================================================================================================




#df.dropna(axis=0,  how='any', inplace=False)

print('Done')
