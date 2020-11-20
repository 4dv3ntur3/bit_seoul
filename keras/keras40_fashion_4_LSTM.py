#2020-11-17 (7일차)
#fashon_mnist -> LSTM
#흑백


from tensorflow.keras.datasets import fashion_mnist

#이미지 분류-> OneHotEncoding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D #maxpooling2d는 들어가도 되고 안 들어가도 됨 필수 아님
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train[0])
# print("y_train[0]: ", y_train[0])

#데이터 구조 확인
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)        
print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2]).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2]).astype('float32')/255.


x_predict = x_train[:10]
y_answer = y_train[:10]


#2. 모델
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activatio n default='relu'
                                        #LSTM의 activation default='tanh'
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid(여자/남자, dead/alive)
                                            #즉 softmax를 사용하려면 OneHotEncoding 해야




#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=512, verbose=1,
          validation_split=0.2, callbacks=[early_stopping])



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)

print("======fashion_lstm=======")
model.summary()

print("loss: ", loss)
print("acc: ", accuracy)


#정답
y_answer = np.argmax(y_answer, axis=1)

#예측값
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)

print("예측값: ", y_predict)
print("정답: ", y_answer)


'''
======fashion_lstm=======
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 64)                23808
_________________________________________________________________
dense (Dense)                (None, 128)               8320
_________________________________________________________________
dense_1 (Dense)              (None, 256)               33024
_________________________________________________________________
dense_2 (Dense)              (None, 512)               131584
_________________________________________________________________
dense_3 (Dense)              (None, 300)               153900
_________________________________________________________________
dense_4 (Dense)              (None, 128)               38528
_________________________________________________________________
dense_5 (Dense)              (None, 64)                8256
_________________________________________________________________
dense_6 (Dense)              (None, 30)                1950
_________________________________________________________________
dense_7 (Dense)              (None, 10)                310
=================================================================
Total params: 399,680
Trainable params: 399,680
Non-trainable params: 0
_________________________________________________________________
loss:  0.5856268405914307
acc:  0.8751000165939331
예측값:  [9 0 0 3 0 2 7 2 5 5]
정답:  [9 0 0 3 0 2 7 2 5 5]
'''