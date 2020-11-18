#2020-11-18 (8일차)
#가중치 load

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist 

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() 


#전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.



#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(30, (2, 2), padding='same', input_shape=(28, 28, 1))) 
model.add(Conv2D(50, (2, 2), padding='valid'))
model.add(Conv2D(120, (3, 3))) 
model.add(Conv2D(200, (2, 2), strides=2))
model.add(Conv2D(30, (2, 2)))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Flatten()) 
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))





#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy']) 


#모델 빼고 가중치만 저장하므로 가중치를 불러올 때 모델은 남겨 둬야 함
#model.load_weights니까 ㅋㅋㅋㅋㅋ
model.load_weights('./save/weight_test02.h5')


#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", result[0])
print("acc: ", result[1])



#정답
# x_predict = x_test[:10]
# y_answer = y_test[:10]
# y_answer = np.argmax(y_answer, axis=1)

#예측값
# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)

# print("예측값: ", y_predict)
# print("정답: ", y_answer)




#결과값
'''
loss:  0.08255356550216675
acc:  0.980400025844574
예측값:  [4 0 9 1 1 2 4 3 2 7]
정답:  [4 0 9 1 1 2 4 3 2 7]

'''

#똑같다
#가중치만 저장하고 싶으면 weights_save
#잘 만든 모델이라고 판단되면 -> 
'''
loss:  0.08255356550216675
acc:  0.980400025844574
'''