
# Multi Layer Perceptron

#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(711, 811), range(100)])
y = np.array(range(101, 201))


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
from tensorflow.keras.models import Sequential #순차적 모델
from tensorflow.keras.layers import Dense #DNN 구조의 Dense모델

model = Sequential()
#model.add(Dense(100, input_dim=3)) #column개수=3개
model.add(Dense(100, input_shape=(3,))) #column개수 = 3개 (스칼라가 3개) 위와 동일
#if (100, 10,3) input shape = (10,3) 제일 앞은 항상 행(전체 데이터 개수) 행 무시 열 우선! 
#이제부턴 shape
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(1)) # output y=1이므로 노드 1개


#3. 컴파일, 훈련
#val_loss로도 많이 검증함! 중요!
#훈련할 때 loss, 아직 모르는 데이터로 loss
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, 
                            validation_split=0.25, 
                            verbose=3)

                            #0은 epoch 안 나옴: 출력에도 시간이 많이 소요됨 대신 중간에 print 넣어서 디버그 비슷한 것만 함 여기까지 출력됐다 이런 느낌
                            #2는 progress bar 안 나옴
                            #default=1
                            #3은 그냥 세부정보 없이 epoch 횟수만 기재

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)
