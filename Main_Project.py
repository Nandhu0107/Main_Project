import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
print("Initial Dataset Shape:", df.shape)

df['date_time'] = pd.to_datetime(df['date_time'])

df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.day
df['month'] = df['date_time'].dt.month
df['day_of_week'] = df['date_time'].dt.dayofweek

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df['is_peak_hour'] = (
    ((df['hour'] >= 7) & (df['hour'] <= 10)) |
    ((df['hour'] >= 16) & (df['hour'] <= 19))
).astype(int)

df['congestion_level'] = pd.qcut(
    df['traffic_volume'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

np.random.seed(42)
df['emergency_flag'] = 0
df.loc[df.sample(frac=0.02).index, 'emergency_flag'] = 1

df = pd.get_dummies(
    df,
    columns=['weather_main', 'weather_description', 'holiday'],
    drop_first=True
)

scaler = MinMaxScaler()
scale_columns = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
df[scale_columns] = scaler.fit_transform(df[scale_columns])

df.drop(columns=['date_time'], inplace=True)

df.to_csv("preprocessed_traffic_data.csv", index=False)

print("Data preprocessing completed successfully")
print("Final Dataset Shape:", df.shape)
