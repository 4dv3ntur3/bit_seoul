#X1 -> Y1, Y2, Y3
#1. 데이터
import numpy as np

x1 = np.array([range(1, 101), range(711, 811), range(100)])


y1 = np.array([range(101, 201), range(311, 411), range(100)])
y2 = np.array([range(501, 601), range(431, 531), range(100, 200)])
y3 = np.array([range(501, 601), range(431, 531), range(100, 200)])


x1 = x1.T
y1 = y1.T
y2 = y2.T
y3 = y3.T

#data 분리 (인자 3개까지 가능하므로 둘 중 하나에 y3 추가해도 됨)
from sklearn.model_selection import train_test_split
x1_train, x1_test = train_test_split(
    x1, shuffle=True, train_size=0.7
)

y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    y1, y2, y3, shuffle=True, train_size=0.7
)


#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#두 모델은 각각 별개. 서로 영향을 끼치지 않는다.
#모델 1
input1 = Input(shape=(3,))
dense1_1 = Dense(10, activation='relu', name="dense1_1")(input1) 
dense1_2 = Dense(5, activation='relu', name="dense1_2")(dense1_1) 
dense1_3 = Dense(7, activation='relu', name="dense1_3")(dense1_2)
output__1 = Dense(3)(dense1_3) #y도 100, 3임 
model1 = Model(inputs=input1, outputs=output__1)

#model1.summary()


'''
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
'''


################ ouput 모델 구성 (분기)
output1 = Dense(30)(output__1)
output1 = Dense(7)(output1)
output1 = Dense(3, name="y1")(output1)

output2 = Dense(15)(output__1)
output2_1 = Dense(14)(output2)
output2_3 = Dense(11)(output2_1)
output2_4 = Dense(3, name="y2")(output2_3)

output3 = Dense(30)(output__1)
output3_1 = Dense(15)(output3)
output3_3 = Dense(15)(output3_1)
output3_4 = Dense(3, name="y3")(output3_3)


#모델 정의
model = Model(inputs=input1, outputs=[output1, output2_4, output3_4])

model.summary()

#concatenate는 연산하지 않는다 


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit(
    x1_train,
    [y1_train, y2_train, y3_train],
    epochs=100, batch_size=8,
    validation_split=0.25,
    verbose=1
)


#4. 평가
result = model.evaluate([x1_test], [y1_test, y2_test, y3_test], 
                batch_size=8)

#수치가 5개인 이유: 전체 loss, 각각 두 아웃풋들의 loss 및 mse
#전체 loss = 각 아웃풋의 loss+mse 합 

print("result: ", result)

#model에 맞춰서 input을 주어야 한다
y_pred_1, y_pred_2, y_pred_3 = model.predict(x1_test)


from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다

print("\nRMSE_1: ", RMSE(y1_test, y_pred_1))
print("RMSE_2: ", RMSE(y2_test, y_pred_2))
print("RMSE_3: ", RMSE(y3_test, y_pred_3), "\n")


# R2는 함수 제공
from sklearn.metrics import r2_score
print("R2_1: ", r2_score(y1_test, y_pred_1))
print("R2_2: ", r2_score(y2_test, y_pred_2))
print("R2_3: ", r2_score(y3_test, y_pred_3))