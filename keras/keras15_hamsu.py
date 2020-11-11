# Multi Layer Perceptron

#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(711, 811), range(100)])
y = np.array([range(101, 201)])


#데이터 shape 확인
print(x.shape)
print(y.shape)

#맞춰 주기
x = x.transpose()
y = y.reshape(100, )



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

#행 무시, 열 우선
#data의 특성은 "열"에 의해 결정된다
#특성=feature=column=열

#최종목표: y1, y2, y3 = w1x1 + w2x2 + w3x3 + b (y도 3개)

#2. 모델 구성
# from keras.models 이렇게 해도 되긴 한다
# 단, keras 쓰려면 tensorflow를 백엔드에 깔고 올라가야 하므로 keras.models로 데려오면 조금 느려진다 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

'''
#dense 층에서는 activaiton default=linear
model = Sequential()
#model.add(Dense(100, input_dim=3)) #column개수=3개
model.add(Dense(5, input_shape=(3,), activation='relu')) #column개수 = 3개 (스칼라가 3개) 위와 동일
#if (100, 10,3) input shape = (10,3) 제일 앞은 항상 행(전체 데이터 개수) 행 무시 열 우선! 
#이제부턴 shape
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1)) # output y=1이므로 노드 1개

#sequential = 최상단에서 정의 후 add로 위->아래로 엮었다


#행 무시니까 100, 3 input시 none, 3으로 들어가게 됨
#연산되는(연결되는) 선 하나가 parameter 1개
'''

# 행 크기 맞춰 줘야 한다
#activation function
#모든 layer마다 존재, dense는 원래 기본적으로 linear지만 지금은 성능 고려해서 relu
#relu 쓰면 85점 이상 (평타)
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1) #상단의 input layer를 사용하겠다
dense2 = Dense(4, activation='relu')(dense1) 
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3) #선형회귀이므로 마지막 activation은 linear여야 한다

model = Model(inputs=input1, outputs=output1)
#함수형은 일단 쌓고 아래에서 모델 정의


model.summary()

# param_number = output_channel_number * (input_channel_number + 1)
# (이전 층의 노드 + 1) * 이번 층 노드 
# y = wx + "b" 따라서 다음층 노드 수만큼 b도 입력 (모든 레이어에 강림함)



'''
#3. 컴파일, 훈련
#val_loss로도 많이 검증함! 중요!
#훈련할 때 loss, 아직 모르는 데이터로 loss
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, 
                            validation_split=0.25, 
                            verbose=1, batch_size=1)

                            #0은 epoch 안 나옴: 출력에도 시간이 많이 소요됨 대신 중간에 print 넣어서 디버그 비슷한 것만 함 여기까지 출력됐다 이런 느낌
                            #2는 progress bar 안 나옴
                            #default=1
                            #3은 그냥 세부정보 없이 epoch 횟수만 기재

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)
'''


