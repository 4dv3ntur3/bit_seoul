#2020-12-04
#embedding + 학습용 예제

from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#단어사전의 개수 
words = 10000


#1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = words, #단어 사전의 개수. 조절 가능
    test_split=0.2
)

print(x_train.shape, x_test.shape) #(8982,) #(2246,)   #8982개의 문장
print(y_train.shape, y_test.shape) #(8982,) #(2246,) y는 46개의 영역 


print(x_train[0])
print(y_train[0])
#tokenizing 되어 있음 

print(len(x_train[0]))      #87
print(len(x_train[11]))     #59


#y의 카테고리 개수 출력
category = np.max(y_train) + 1
print("카테고리: ", category) #카테고리:  46 (softmax)

#신문기사 맞추기 

#y의 유니크한 값을 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]


#실습: embedding 모델 구성 + 완료 + 끝



#padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

#data 개수가 많으므로 maxlen 사용해서 손실 감수하기 
x_train = pad_sequences(x_train, maxlen=1000, padding='pre')
x_test = pad_sequences(x_test, maxlen=1000, padding='pre')

print(x_train.shape) #(8982, 2376)
print(x_test.shape) #(2246, 1032)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(8982, 46)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(words, 64, input_length=x_train.shape[1]))
model.add(Flatten()) #output이 3차원임
model.add(Dense(46, activation='softmax'))

model.summary()


model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['acc'])

model.fit(x_train, y_train, epochs=30)

acc = model.evaluate(x_test, y_test)[1]
print("acc: ", acc)


'''
acc:  0.703027606010437
'''