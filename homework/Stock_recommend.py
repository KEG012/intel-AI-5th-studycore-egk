#%%
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

kospi_200 = fdr.StockListing('KOSPI')

df = fdr.DataReader('005930', '2021-01-01', '2024-09-19')

# 이동평균선 계산
df['MA20'] = df['Close'].rolling(window=20).mean()  # 20일 이동평균선
df['MA60'] = df['Close'].rolling(window=60).mean()  # 60일 이동평균선

# 단기 이동평균이 장기 이동평균을 상향 돌파하면 매수 신호
df['Buy_Signal'] = np.where(df['MA20'] > df['MA60'], 1, 0)

# NaN 값 제거 (이동평균선을 계산하면서 발생한 NaN 처리)
df = df.dropna()

# features, labels
features = df[['MA20', 'MA60']].values
labels = df['Buy_Signal'].values

# data regulation
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# GRU input (3D)
features = features.reshape((features.shape[0], 1, features.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# model construction
model = Sequential()
model.add(GRU(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(GRU(50))
model.add(Dropout(0.1))
model.add(Dense(1, activation='elu'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# stock recommend
predictions = model.predict(X_test)
predicted_signals = (predictions > 0.5).astype(int).flatten()

# if Buy_Signal == 1 recommend
recommended_stocks = df[df['Buy_Signal'] == 1]
print(recommended_stocks[['Close', 'MA20', 'MA60']])

# show plot
plt.figure(figsize=(14,7))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['MA20'], label='20-Day MA')
plt.plot(df['MA60'], label='60-Day MA')
plt.legend()
plt.show()