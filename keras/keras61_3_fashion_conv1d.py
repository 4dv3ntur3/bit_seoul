#2020-11-19
#LSTM vs Conv1d: fashon


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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

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
model.add(Dense(10, activation='softmax')) #ouput 



#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1,
          validation_split=0.2, callbacks=[early_stopping])



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=100)

print("======fashion_conv1d=======")


print("loss: ", loss)
print("acc: ", accuracy)


#정답
y_answer = np.argmax(y_answer, axis=1)

#예측값
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)

print("예측값: ", y_predict)
print("정답: ", y_answer)

model.summary()


'''
======fashion_conv1d=======
loss:  0.7468854784965515
acc:  0.9006999731063843
예측값:  [9 0 0 3 0 2 7 2 5 5]
정답:  [9 0 0 3 0 2 7 2 5 5]
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 28, 256)           21760
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 28, 256)           196864
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 14, 256)           0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 14, 128)           98432
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 14, 128)           49280
_________________________________________________________________
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 7, 512)            197120
_________________________________________________________________
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
dense_1 (Dense)              (None, 10)                10250
=================================================================
Total params: 1,164,682
Trainable params: 1,164,682
Non-trainable params: 0
_________________________________________________________________
'''