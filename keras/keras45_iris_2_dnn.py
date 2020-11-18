#2020-11-18 (8일차)
#다중분류
#꽃잎과 줄기를 보고 어떤 꽃인지 판별하는 데이터 -> DNN
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


print(x.shape) #(150, 4)
print(y.shape) #(150,)

# #OneHotEncoding (다중분류)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


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


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_train.shape[1]).astype('float32')/255.

print(x_train.shape)


#OneHotEncoding (다중분류)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_predict = x_train[:10]
y_answer = y_train[:10]


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activatio n default='relu'
                                        #LSTM의 activation default='tanh'
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid(여자/남자, dead/alive)
                                            #즉 softmax를 사용하려면 OneHotEncoding 해야




#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=1000, batch_size=32
)



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)

print("=======iris_dnn=======")
model.summary()
print("loss: ", loss)
print("acc: ", accuracy)


# #정답
# y_answer = np.argmax(y_answer, axis=1)

# #예측값
# y_predict = model.predict(x_predict)
# y_predict = np.argmax(y_predict, axis=1)

# print("예측값: ", y_predict)
# print("정답: ", y_answer)


'''
=======iris_dnn=======
Model: "sequential"
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                320
_________________________________________________________________
dense_1 (Dense)              (None, 512)               33280
_________________________________________________________________
dropout (Dropout)            (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328
_________________________________________________________________
dense_3 (Dense)              (None, 128)               32896
_________________________________________________________________
dense_4 (Dense)              (None, 300)               38700
_________________________________________________________________
dense_5 (Dense)              (None, 150)               45150
_________________________________________________________________
dense_6 (Dense)              (None, 70)                10570
_________________________________________________________________
dense_7 (Dense)              (None, 3)                 213
=================================================================
Total params: 292,457
Trainable params: 292,457
Non-trainable params: 0
_________________________________________________________________
loss:  0.11479297280311584
acc:  0.9777777791023254
예측값:  [1 0 1 1 0 2 2 1 1 2]
정답:  [1 0 1 1 0 1 2 1 2 2]
'''