#2020-11-19 (9일차)


import numpy as np
import pandas as pd


#iris_ys2.csv 파일을 numpy로 불러오기
data_np = np.loadtxt('./data/csv/iris_ys2.csv', delimiter=',')
print(type(data_np))


#불러온 데이터를 판다스로 저장(*.csv)하시오.
data_pd = pd.DataFrame(data_np)
data_pd.to_csv('./data/csv/iris_ys2_pd.csv')


# print(data_np.shape) #(150, 5)
# print(type(data_np))


#1. 데이터

x = data_np[:, :4]
y = data_np[:, 4:]


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from tensorflow.keras.utils import to_categorical


#전처리
x_train = x_train.reshape(x_train.shape[0], 4, 1, 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 4, 1, 1).astype('float32')/255.

#OneHotEncoding (다중분류)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(4, 1, 1))) #padding 주의!
model.add(Dropout(0.2))


model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) #padding default=valid
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) #padding default=valid
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(3, activation='softmax')) #ouput 
model.summary()


