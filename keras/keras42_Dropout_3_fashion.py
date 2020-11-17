#2020-11-17 (7일차)
#fashon_mnist -> CNN
#Dropout 추가
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

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')/255.


x_predict = x_train[:10]
y_answer = y_train[:10]


#2. 모델
model = Sequential()
model.add(Conv2D(200, (3, 3), padding='same', input_shape=(x_train.shape[1], x_train.shape[2], 1))) #padding 주의!
model.add(Conv2D(180, (2, 2), padding='valid'))
model.add(Conv2D(100, (3, 3), strides=2)) #padding default=valid
model.add(Conv2D(50, (2, 2)))
model.add(Conv2D(30, (2, 2)))
model.add(Conv2D(10, (3, 3)))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Flatten()) 
model.add(Dense(10, activation='relu')) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activation default='relu'
                                        #LSTM의 activation default='tanh'
# model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax')) #label: 0~9 (항상 dataset label 확인)



#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
          validation_split=0.2, callbacks=[early_stopping])



#4. 평가, 예측
print("====Dropout_fasion====")

loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)

print("loss: ", loss)
print("acc: ", accuracy)


#정답
y_answer = np.argmax(y_answer, axis=1)

#예측값
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)

print("예측값: ", y_predict)
print("정답: ", y_answer)
