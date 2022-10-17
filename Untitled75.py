#!/usr/bin/env python
# coding: utf-8

# In[17]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime
from datetime import date 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[18]:


pip install pandas_datareader

# In[19]:


import pandas_datareader.data as web
from pandas import Series, DataFrame
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2019, 12, 31)

# In[20]:


df = web.DataReader("MSFT", 'yahoo', start, end)
df.tail()

# In[21]:


df.head()

# In[22]:


df.isnull().sum()

# In[23]:


close_px = df['Adj Close']
mavg = close_px.rolling(window = 100).mean()
mavg.tail(10)

# In[24]:


df.shape

# In[25]:


df['Close'].hist()

# In[26]:


df['Close'].plot()
plt.xlabel("Date")
plt.ylabel("Close")


# In[27]:


df['Close'].plot(style='.')
plt.title("Scatter plot of Closing Price")
plt.title('Scatter plot of Closing Price',fontsize=20)
plt.show()

# In[28]:


#Test for staionarity
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolling_mean = timeseries.rolling(12).mean()
    rolling_std = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='Black',label='Original')
    plt.plot(rolling_mean, color='Green', label='Rolling Mean')
    plt.plot(rolling_std, color='Red', label = 'Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation',fontsize=20)
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

# In[29]:


test_stationarity(df['Close'])

# In[30]:


from pylab import rcParams
rcParams['figure.figsize'] = 10, 9
df_log = np.log(df.Close)
moving_avg = df.Close.rolling(12).mean()
std_dev = df.Close.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average',fontsize=20)
plt.plot(std_dev, color ="Blue", label = "Standard Deviation")
plt.plot(moving_avg, color="Green", label = "Mean")
plt.legend()
plt.show()

# In[31]:


#Split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,9))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()

# In[ ]:




# In[ ]:




# In[ ]:


auto_arima_model.plot_diagnostics()
plt.show()

# In[ ]:





# In[ ]:



