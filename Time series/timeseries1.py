import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

# reading csv file 
df = pd.read_csv('C:\\Users\\mahim\\OneDrive\\Desktop\\Minor-2\\multiTimeline.csv')
#print(df.head())
print()

# making month column to date-time data type
df.Month = pd.to_datetime(df.Month)

# changing index as month column 
df.set_index('Month' , inplace=True)
#print(df.head())

# ploting all three column diet, gym and finance
#df.plot(figsize=(20,10), linewidth=3, fontsize=20)
#plt.xlabel("Year", fontsize=20)
#plt.show()

# making all three columns different
diet = df[['Diet']]
gym = df[['Gym']]
finance = df[['Finance']]

# checking trend in all three
'''diet.rolling(1).mean().plot(figsize=(7,7), linewidth=5, fontsize=20)
plt.xlabel("Year", fontsize=20)
plt.show()
gym.rolling(1).mean().plot(figsize=(7,7), linewidth=5, fontsize=20)
plt.xlabel("Year", fontsize=20)
plt.show()
finance.rolling(1).mean().plot(figsize=(7,7), linewidth=5, fontsize=20)
plt.xlabel("Year", fontsize=20)
plt.show()'''

# checking seasonality in all three
r1 = seasonal_decompose(diet, model='multiplicative')

'''r1.plot()
plt.show()
r1.seasonal.plot()
plt.show()'''

r2 = seasonal_decompose(gym, model='multiplicative')
'''r2.plot()
plt.show()'''

r3 = seasonal_decompose(finance, model='multiplicative')
'''r3.plot()
plt.show()'''

diet.plot()
plt.show()

# removing trend
#diet = [diet[i]-r1.trend[i] for i in range(0, len(diet))]
#diet.plot()
#plt.show()

# removing seasonality
r1_without_seasonal = diet['Diet']/r1.seasonal
'''diet.plot()
r1_without_seasonal.plot()
plt.show()'''

r2_without_seasonal = gym['Gym']/r2.seasonal
'''gym.plot()
r2_without_seasonal.plot()
plt.show()'''

r3_without_seasonal = finance['Finance']/r3.seasonal
'''finance.plot()
r3_without_seasonal.plot()
plt.show()'''

# making new dataframe without seasonality
df_new = pd.DataFrame([r1_without_seasonal,r2_without_seasonal,r3_without_seasonal]).transpose()
df_new.rename(columns={0:'diet',1:'gym',2:'finance'}, inplace =True)
#print(df_new.head())
#df_new.to_csv('C:\\Users\\mahim\\OneDrive\\Desktop\\Minor-2\\result1.csv')
