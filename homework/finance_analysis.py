#%%
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout

# 범위를 0 ~ 1 로 normalized
def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

# 삼성전자 주식 데이터
df = fdr.DataReader('005930', '2022-01-01', '2024-09-19')
dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]

# print(dfx)
# print(dfy)

x = dfx.values.tolist() # open, high, log, volume <- A M
y = dfy.values.tolist() # close data

# print(x)
# print(y)

#ex) 1월 1일 ~ 1월 10일까지의 OHLV 데이터로 1월 11일 종가 (Close) 예측
#ex) 1월 2일 ~ 1월 11일까지의 OHLV 데이터로 1월 12일 종가 (Close) 예측
window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]
    # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)
    
# print(data_x)
# print(data_y)
    
train_size = int(len(data_y)*0.7)
val_size = int(len(data_y)*0.2)
test_size = int(len(data_y)*0.1)
# print(train_size)
# print(val_size)
# print(test_size)
train_data_x = np.array(data_x[0:train_size])
train_data_y = np.array(data_y[0:train_size])
# print(train_data_x)
# print(train_data_y)
val_data_x = np.array(data_x[train_size:train_size + val_size])
val_data_y = np.array(data_y[train_size:train_size + val_size])
# print(val_data_x)
# print(val_data_y)
test_data_x = np.array(data_x[train_size + val_size:train_size + val_size + test_size])
test_data_y = np.array(data_y[train_size + val_size:train_size + val_size + test_size])

print('훈련 데이터의 크기 :', train_data_x.shape, train_data_y.shape)
print('검증 데이터의 크기 :', val_data_x.shape, val_data_y.shape)
print('테스트 데이터의 크기 :', test_data_x.shape, test_data_y.shape)

# RNN Model Construction
model_in_RNN = Input(shape = (10, 4))
model_1 = SimpleRNN(units = 40, activation = 'elu', return_sequences = True)(model_in_RNN)
model_1 = Dropout(0.1)(model_1)
model_1 = SimpleRNN(units = 40, activation = 'elu')(model_1)
model_1 = Dropout(0.1)(model_1)
model_1 = Dense(units = 1)(model_1)
model_RNN = Model(inputs = model_in_RNN, outputs = model_1)
model_RNN.summary()

model_RNN.compile(optimizer = 'adam', loss = 'mean_squared_error')
history1 = model_RNN.fit(train_data_x, train_data_y, validation_data = (val_data_x, val_data_y),
                    epochs = 70, batch_size = 30)



# GRU Model Construction
model_in_GRU = Input(shape = (10, 4))
model_2 = GRU(units = 40, activation = 'elu', return_sequences = True)(model_in_GRU)
model_2 = Dropout(0.1)(model_2)
model_2 = GRU(units = 40, activation = 'elu')(model_2)
model_2 = Dropout(0.1)(model_2)
model_2 = Dense(units = 1)(model_2)
model_GRU = Model(inputs = model_in_GRU, outputs = model_2)
model_GRU.summary()

model_GRU.compile(optimizer = 'adam', loss = 'mean_squared_error')
history2 = model_GRU.fit(train_data_x, train_data_y, validation_data = (val_data_x, val_data_y),
              epochs = 70, batch_size = 30)

# LSTM Model Construction
model_in_LSTM = Input(shape = (10, 4))
model_3 = LSTM(units = 40, activation = 'elu', return_sequences = True)(model_in_LSTM)
model_3 = Dropout(0.1)(model_3)
model_3 = GRU(units = 40, activation = 'elu')(model_3)
model_3 = Dropout(0.1)(model_3)
model_3 = Dense(units = 1)(model_3)
model_LSTM = Model(inputs = model_in_LSTM, outputs = model_3)
model_LSTM.summary()

model_LSTM.compile(optimizer = 'adam', loss = 'mean_squared_error')
history3 = model_LSTM.fit(train_data_x, train_data_y, validation_data = (val_data_x, val_data_y),
              epochs = 70, batch_size = 30)

# test data predict
RNN_predict = model_RNN.predict(test_data_x)
GRU_predict = model_GRU.predict(test_data_x)
LSTM_predict = model_LSTM.predict(test_data_x)

# print(RNN_predict.shape)
# print(GRU_predict.shape)
# print(LSTM_predict.shape)

# test data reshape (2D -> 1D)
RNN_predict = RNN_predict.reshape(-1)
GRU_predict = GRU_predict.reshape(-1)
LSTM_predict = LSTM_predict.reshape(-1)
actual = test_data_y.reshape(-1)

# print(RNN_predict.shape)
# print(GRU_predict.shape)
# print(LSTM_predict.shape)

# print(RNN_predict)
# print(GRU_predict)
# print(LSTM_predict)

# data plot
plt.figure(figsize=(14, 8))
# Actual
plt.plot(actual, label='Actual', color='red')
# RNN_predict
plt.plot(RNN_predict, label='predicted (RNN)', linestyle='--', color='blue')
# GRU_predict
plt.plot(GRU_predict, label='predicted (GRU)', linestyle='--', color='yellow')
# LSTM_predict
plt.plot(LSTM_predict, label='predicted (LSTM)', linestyle='--', color='green')

# graph detail set
plt.title('SEC stock Price Prediction')
plt.xlabel('time')
plt.ylabel('stock Price')
plt.legend()
plt.grid(True)  

# show graph
plt.show()

'''
plt.title('Financial Data Comparison')
plt.plot()
'''