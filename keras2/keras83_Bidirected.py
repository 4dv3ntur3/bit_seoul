#2020-12-07
#시계열 데이터의 경우 연산을 -> , <- 두 번 하면 좋지 않을까? -> bidirectional(layer임)
#딥러닝에선 무조건 옳다가 없다. 경우의 수가 늘었을 뿐.
#클래스화하고 함수화해서 사용할 것 
#클래스로 싸서 발표하기 




from tensorflow.keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#단어사전의 개수 
words = 10000

#1. 데이터
#imdb에는 test_split가 없음 
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=words
)


print(x_train.shape, x_test.shape) #(25000, 100)
print(y_train.shape, y_test.shape) #(25000,) 


print(x_train[0])
print(y_train[0])
#tokenizing 되어 있음 

print(len(x_train[0]))      
print(len(x_train[11]))     #59


#y의 카테고리 개수 출력
category = np.max(y_train) + 1
print("카테고리: ", category) 

#신문기사 맞추기 

#y의 유니크한 값을 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)




#실습: embedding 모델 구성 + 완료 + 끝

#padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

#data 개수가 많으므로 maxlen 사용해서 손실 감수하기 
x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')

print(x_train.shape) #(25000, 1000)
print(x_test.shape) #(25000, 1000)

print(y_train.shape) #(25000,)
print(y_test.shape) #(25000,)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, Bidirectional

model = Sequential()
model.add(Embedding(words, 64, input_length=x_train.shape[1]))
model.add(Bidirectional(LSTM(128, input_shape=(5, 1)))) #그냥 LSTM 했을 때보다 parameters 2배 & output shape도 두 배 
# model.add(Conv1D(128, 2))
# model.add(Conv1D(64, 2))
# model.add(Flatten()) #output이 3차원임
model.add(Dense(1, activation='sigmoid')) #긍정 or 부정 

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 100, 64)           640000
_________________________________________________________________
lstm (LSTM)                  (None, 100, 128)          98816
_________________________________________________________________
conv1d (Conv1D)              (None, 99, 128)           32896
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 98, 64)            16448
_________________________________________________________________
flatten (Flatten)            (None, 6272)              0
_________________________________________________________________
dense (Dense)                (None, 1)                 6273
=================================================================
Total params: 794,433
Trainable params: 794,433
Non-trainable params: 0
_________________________________________________________________


'''
'''
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['acc'])

model.fit(x_train, y_train, epochs=30)

acc = model.evaluate(x_test, y_test)[1]
print("acc: ", acc)
'''
