#2020-11-17 (7일차)
#cifar-100 -> CNN
#color이므로 channel=3

from tensorflow.keras.datasets import cifar100

#이미지 분류-> OneHotEncoding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout
from tensorflow.keras.layers import Flatten, MaxPooling2D #maxpooling2d는 들어가도 되고 안 들어가도 됨 필수 아님
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#데이터 구조 확인
# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)


#1. 데이터 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3).astype('float32')/255.


x_predict = x_train[:10]
y_answer = y_train[:10]


#2. 모델
model = Sequential()

#input layer
model.add(Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3))) #padding 주의!


#hidden layer
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), strides=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))


model.add(Flatten()) 
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.5))

#output layer
model.add(Dense(100, activation='softmax')) #ouput 맞춰 줘야 






#3. 컴파일, 훈련
#patience는 대개 10?
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=1,
          validation_split=0.2, callbacks=[early_stopping])



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=100)

print("======cifar-100_CNN=======")
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

model.summary()

