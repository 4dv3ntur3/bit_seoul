#2020-11-19
#LSTM vs Conv1d: breast_cancer



import numpy as np
from sklearn.datasets import load_iris

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout



#1. 데이터 
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)



#전처리
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)




# print(x_train.shape)


x_predict = x_train[30:40]
y_answer = y_train[30:40]


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

model.add(Conv1D(512, 3, padding='same', activation='relu'))
model.add(Conv1D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) #ouput 



#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=100, batch_size=32
)



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)

print("=======cancer_conv1d=======")
print("loss: ", loss)
print("acc: ", accuracy)


#정답

#예측값
y_predict = model.predict(x_predict)

print("예측값: ", y_predict)
print("정답: ", y_answer)


model.summary()

'''
=======cancer_conv1d=======
loss:  0.10820714384317398
acc:  0.9707602262496948
예측값:  [[9.9982721e-01]
 [9.7370398e-01]
 [3.8489670e-06]
 [3.4278909e-15]
 [9.9999213e-01]
 [9.9928004e-01]
 [9.9983585e-01]
 [9.9999988e-01]
 [1.2035067e-08]
 [9.9996686e-01]]
정답:  [1 1 0 0 1 1 1 1 0 1]
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 30, 256)           1024
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 30, 256)           196864
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 15, 256)           0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 15, 128)           98432
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 15, 128)           49280
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 7, 128)            0
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 7, 512)            197120
conv1d_5 (Conv1D)            (None, 7, 128)            196736
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 3, 128)            0
_________________________________________________________________
flatten (Flatten)            (None, 384)               0
_________________________________________________________________
dense (Dense)                (None, 1024)              394240
_________________________________________________________________
dropout (Dropout)            (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 1025
=================================================================
Total params: 1,134,721
Trainable params: 1,134,721
Non-trainable params: 0
_________________________________________________________________
'''