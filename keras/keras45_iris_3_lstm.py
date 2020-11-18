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
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]))) #꼭 28, 28, 1일 필요는 없음 #뭔가 시계열 같은 데이터라고 판단이 되면 몇 개씩 자를지 생각할 수도 있음
model.add(Dense(128, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(750, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(30, activation='relu')) 


model.add(Dense(3, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid(여자/남자, dead/alive)
                                            #즉 softmax를 사용하려면 OneHotEncoding 해야




#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=1000, batch_size=1
)



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=1)

print("=======iris_lstm=======")

model.summary()
print("loss: ", loss)
print("acc: ", accuracy)



#정답

#예측값
# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)

# print("예측값: ", y_predict)
# print("정답: ", y_test)



'''
=======iris_lstm=======
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 64)                16896
_________________________________________________________________
dense (Dense)                (None, 128)               8320
_________________________________________________________________
dense_1 (Dense)              (None, 300)               38700
_________________________________________________________________
dropout (Dropout)            (None, 300)               0
_________________________________________________________________
dense_2 (Dense)              (None, 750)               225750
_________________________________________________________________
dense_3 (Dense)              (None, 1024)              769024
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_4 (Dense)              (None, 512)               524800
_________________________________________________________________
dense_5 (Dense)              (None, 256)               131328
_________________________________________________________________
dense_6 (Dense)              (None, 90)                23130
_________________________________________________________________
dense_7 (Dense)              (None, 30)                2730
_________________________________________________________________
dense_8 (Dense)              (None, 3)                 93
=================================================================
Total params: 1,740,771
Trainable params: 1,740,771
Non-trainable params: 0
_________________________________________________________________
loss:  0.3195541203022003
acc:  0.9777777791023254
'''


'''
=======iris_lstm=======
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 64)                16896
_________________________________________________________________
dense (Dense)                (None, 128)               8320
_________________________________________________________________
dense_1 (Dense)              (None, 300)               38700
_________________________________________________________________
dense_2 (Dense)              (None, 750)               225750
_________________________________________________________________
dropout (Dropout)            (None, 750)               0
_________________________________________________________________
dense_3 (Dense)              (None, 1024)              769024
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_4 (Dense)              (None, 512)               524800
_________________________________________________________________
dense_5 (Dense)              (None, 256)               131328
_________________________________________________________________
dense_6 (Dense)              (None, 90)                23130
_________________________________________________________________
dense_7 (Dense)              (None, 30)                2730
_________________________________________________________________
dense_8 (Dense)              (None, 3)                 93
=================================================================
Total params: 1,740,771
Trainable params: 1,740,771
Non-trainable params: 0
_________________________________________________________________
loss:  0.18281465768814087
acc:  0.9777777791023254
'''


'''
=======iris_lstm=======
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 64)                16896
_________________________________________________________________
dense (Dense)                (None, 128)               8320
_________________________________________________________________
dense_1 (Dense)              (None, 300)               38700
_________________________________________________________________
dense_2 (Dense)              (None, 750)               225750
_________________________________________________________________
dropout (Dropout)            (None, 750)               0
_________________________________________________________________
dense_3 (Dense)              (None, 1024)              769024
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_4 (Dense)              (None, 512)               524800
_________________________________________________________________
dense_5 (Dense)              (None, 256)               131328
_________________________________________________________________
dense_6 (Dense)              (None, 90)                23130
_________________________________________________________________
dense_7 (Dense)              (None, 30)                2730
_________________________________________________________________
dense_8 (Dense)              (None, 3)                 93
=================================================================
Total params: 1,740,771
Trainable params: 1,740,771
Non-trainable params: 0
_________________________________________________________________
loss:  0.14333532750606537
acc:  0.9333333373069763
'''