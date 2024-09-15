import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

for i, str in enumerate(df.columns):
    print(i, ': ', str)

# #92.5% predicting productivity loss
# drop_cols = [0,1,2,3,4,5,6,7,8,12,25,30]

# df.drop(df.columns[drop_cols],axis=1,inplace=True)
df = df[['Platform','DeviceType','Total Time Spent','Watch Reason','Addiction Level','ProductivityLoss']]
df.to_csv('cleaned_data.csv', index=False)  

pd.set_option('display.max_columns', 500)

print(df.head(2))    

# print("PLATFORM ***************************")
# print(df['Platform'].unique())

# print("DEVICE TYPE ***************************")
# print(df['DeviceType'].unique())

# print("WATCH REASON ***************************")
# print(df['Watch Reason'].unique())