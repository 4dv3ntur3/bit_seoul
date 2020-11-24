#2020-11-23 (11일차)
#data의 중요성
#Y data 전처리: keras

import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical 



wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)
y = wine['quality']
x = wine.drop('quality', axis=1) #quality를 뺀 나머지가 x

print(x.shape) #4898, 11
print(y.shape) #4898,


import numpy as np

#데이터 전처리의 일환
newlist = []
for i in list(y):
    if i<=2: #4
        newlist +=[0] # 4보다 작으면 labeling = 0

    elif i<=7: #7
        newlist +=[1] 
    
    else: 
        newlist +=[2]


# 3부터 9까지의 데이터 분포, 5, 6, 7이 제일 많았다
# 3, 4, 5, 6, 7, 8, 9로 갈라서 본다 -> 0 1 2로 됨
# data 조작 아닙니까? 조작일 수도 있지만 전처리일 수도 있다
# wine의 품질 판단 데이터셋. wine의 등급이 3~9까지니까 맞추는 거였는데, 조절해서 0~2의 3단계로 줄임.


y = newlist

#2. 모델 만든 거 잇기



#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8
)

scaler = StandardScaler()
scaler.fit(x_train)

scaler.transform(x_train)
scaler.transform(x_test)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델(일단은 dafault만 미세 조정은 추후에)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(x_train.shape[1],))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activatio n default='relu'
                                        #LSTM의 activation default='tanh'
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #2진 분류면 one hot encoding 필요 없음. 3진 분류는 해야 됨
model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[es])



#4. 평가, 예측


# y_predict = model.predict(x_test) #y_data가 나오는지 보면 된다 


# print(x_test, '의 예측 결과: ', y_predict)

acc_1 = model.evaluate(x_test, y_test)
print('model.evaluate: ', acc_1) 


'''

keras ver.

model.evaluate:  [0.2634871304035187, 0.9255102276802063]

model.evaluate:  [0.1419011801481247, 0.968367338180542]


'''

