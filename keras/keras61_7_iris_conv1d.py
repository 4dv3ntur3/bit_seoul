#2020-11-19
#LSTM vs Conv1d: iris


#2020-11-18 (8일차)
#iris -> LSTM
#꽃잎과 줄기를 보고 어떤 꽃인지 판별하는 데이터, 다중분류
#x column=4 y label:1

import numpy as np
from sklearn.datasets import load_iris

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout




#1. 데이터 

#데이터 구조 확인

dataset = load_iris() #data(X)와 target(Y)으로 구분되어 있다
x = dataset.data
y = dataset.target


# print(x.shape) #(150, 4)
# print(y.shape) #(150,)



#전처리
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], 1)

# print(x_train.shape)


#OneHotEncoding (다중분류)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout

model = Sequential()
model.add(Conv1D(256, 3, activation='relu', padding='same', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Conv1D(256, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(Conv1D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(Conv1D(64, 3, activation='relu', padding='same'))
# model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax')) #ouput 



#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=100, batch_size=1
)



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=1)

print("=======iris_conv1d=======")


print("loss: ", loss)
print("acc: ", accuracy)
model.summary()



#정답

#예측값
# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)

# print("예측값: ", y_predict)
# print("정답: ", y_test)



'''
=======iris_conv1d=======
loss:  0.14945511519908905
acc:  0.9333333373069763
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 4, 256)            1024
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 4, 256)            196864
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 2, 256)            0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 2, 128)            98432
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 2, 128)            49280
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 1, 128)            0
conv1d_4 (Conv1D)            (None, 1, 64)             24640
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 1, 64)             12352
_________________________________________________________________
flatten (Flatten)            (None, 64)                0
_________________________________________________________________
dense (Dense)                (None, 1024)              66560
_________________________________________________________________
dropout (Dropout)            (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 3075
=================================================================
Total params: 452,227
Trainable params: 452,227
Non-trainable params: 0
_________________________________________________________________
'''