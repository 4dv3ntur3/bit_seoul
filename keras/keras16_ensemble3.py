#X1, X2 -> Y1
#1. 데이터
import numpy as np

x1 = np.array([range(1, 101), range(711, 811), range(100)])
x2 = np.array((range(4, 104), range(761, 861), range(100)))


y1 = np.array([range(101, 201), range(311, 411), range(100)])


x1 = x1.T
x2 = x2.T
y1 = y1.T


#data 분리 (인자 3개까지 가능하므로 둘 중 하나에 y3 추가해도 됨)
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, shuffle=True, train_size=0.7
)


# import random
# #y3 
# y3_train = y3[:70]
# y3_test = y3[70:]
# random.shuffle(y3_train)
# random.shuffle(y3_test)


#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#두 모델은 각각 별개. 서로 영향을 끼치지 않는다.
#모델 1
input1 = Input(shape=(3,))
dense1_1 = Dense(300, activation='relu', name="dense1_1")(input1) 
dense1_2 = Dense(50, activation='relu', name="dense1_2")(dense1_1) 
dense1_3 = Dense(70, activation='relu', name="dense1_3")(dense1_2)
output1 = Dense(3)(dense1_3) #y도 100, 3임 
model1 = Model(inputs=input1, outputs=output1)

#model1.summary()

#모델 2
input2 = Input(shape=(3,))
dense2_1 = Dense(300, activation='relu', name="dense2_1")(input2) 
dense2_2 = Dense(50, activation='relu', name="dense2_2")(dense2_1) 
dense2_3 = Dense(70, activation='relu', name="dense2_3")(dense2_2)
output2 = Dense(3)(dense2_3) 
model2 = Model(inputs=input2, outputs=output2)

#model2.summary()



#모델 병합, concatenate
#성능 다 똑같지만, 대문자와 소문자는 사용 방법이 틀리다

from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import Concatenate, concatenate
# from keras.layers import Concatenate, concatenate

#2개 이상 합칠 땐 list 사용*****
#밑에서 합쳐진다 
#함수형 모델
#middle1 = Dense(7)(middle1)
#middle1 = Dense(11)(middle1)
#이래도 된다! 

#merge1 = concatenate([output1, output2])
#class니까 생성자를
#Concatenate(axis=1)
merge1 = Concatenate()([output1, output2])
middle1 = Dense(30)(merge1)
middle1 = Dense(7)(middle1)
middle1 = Dense(11)(middle1)



################ ouput 모델 구성 (분기)
output1 = Dense(300)(middle1)
output1 = Dense(70)(output1)
output1 = Dense(3, name="y1")(output1)

# output2 = Dense(15)(middle1)
# output2_1 = Dense(14)(output2)
# output2_3 = Dense(11)(output2_1)
# output2_4 = Dense(3, name="y2")(output2_3)

# output3 = Dense(30)(middle1)
# output3_1 = Dense(15)(output3)
# output3_3 = Dense(15)(output3_1)
# output3_4 = Dense(3, name="y3")(output3_3)


#모델 정의
model = Model(inputs=[input1, input2], outputs=[output1])

model.summary()

#concatenate는 연산하지 않는다 


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit(
    [x1_train, x2_train],
    [y1_train],
    epochs=1000, batch_size=8,
    validation_split=0.25,
    verbose=1
)


#4. 평가
result = model.evaluate([x1_test, x2_test], y1_test, 
                batch_size=8)

#수치가 5개인 이유: 전체 loss, 각각 두 아웃풋들의 loss 및 mse
#전체 loss = 각 아웃풋의 loss+mse 합 

print("result: ", result)

#model에 맞춰서 input을 주어야 한다
y_predict = model.predict([x1_test, x2_test])


# RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다
print("RMSE: ", RMSE(y1_test, y_predict))

# R2는 함수 제공
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print("R2: ", r2)


#과제: R2 최대한 증가 (튜닝 잘해 놓기)