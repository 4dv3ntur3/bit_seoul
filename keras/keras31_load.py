#2020-11-13 (5일차)
#모델 로드: 로드한 모델에 지정된 input_shape와 내가 넣을 데이터의 input_shape가 다를 때

import numpy as np

#1. 데이터

dataset = np.array(range(1,101))
size = 5

#데이터 전처리 
def split_x(seq, size):
    aaa = [] #는 테스트
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]

        #aaa.append 줄일 수 있음
        #소스는 간결할수록 좋다
        # aaa.append([item for item in subset])
        aaa.append(subset)
        
        
        
    # print(type(aaa))
    return np.array(aaa)


dataset = split_x(dataset, size)


#dataset[:, 0:4]
#dataset[:, 4]
#shape 확인하고 print한 다음 주석으로 적어 두기 

x = dataset[0:100, 0:4]
y = dataset[0:100, 4]


x = x.reshape(x.shape[0], 4, 1)

#차원과는 관계없이 비례에 맞춰서 잘라 준다!
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True
)


#모델을 구성하시오.
from tensorflow.keras.models import load_model #Sequential 없이도 돌아감! load_model에서 같이 당겨온다 
from tensorflow.keras.layers import Dense, LSTM #LSTM도 layer



# 모델 불러오기
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input



#기존 모델에 커스터마이징하기
# ValueError: All layers added to a Sequential model should have unique names. 
# Name "dense" is already the name of a layer in this model. 
# Update the `name` argument to pass a unique name.    
# name 지정

# model.add(Dense(10, name="king1"))


# # # 방법1) custom_objects -> 실패... 안 바뀌는 듯 
# model = load_model('./save/keras26_model.h5', custom_objects={'input_shape':(4,1)})
# model.summary()
# m2 = Model(inputs=[model.input], outputs=[model.output])
# m2.summary()
# # model.add(Dense(10, activation='relu', name='new1'))
# # model.add(Dense(1, activation='relu', name='new2'))

# 방법2) 함수형으로 덮어씌우기
model = load_model('./save/keras26_model.h5')

#input layer를 날린다
model.layers.pop(0)

input1 = Input(shape=(4, 1))
dense = model(input1)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)

model.summary() #커스터마이징 할 때마다 모델 구조 살펴보고, 그대로 코드 베껴오고 추가해서 쓸 순 x



#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.3,
    epochs=1000, batch_size=10
)



#4. 평가, 예측
#101.03
loss, mse = model.evaluate(x_test, y_test)
print(loss, mse)


x_predict = np.array([97, 98, 99, 100])
x_predict = x_predict.reshape(1, 4, 1)

y_predict = model.predict(x_predict)
print("예측값: ", y_predict)
