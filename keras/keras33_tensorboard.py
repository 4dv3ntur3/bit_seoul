#2020-11-13 (5일차)
#시각화(그래프): tensorboard
#graph 폴더 생성

#cmd -> d: -> cd graph -> dir /w (디렉토리 확인) -> tensorboard --logdir=.
#브라우저 켜서 http://localhost:6006/


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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM #LSTM도 layer


model = Sequential()
model.add(LSTM(30, activation='relu', input_length=4, input_dim=1)) # *****
model.add(Dense(50, activation='relu')) #default activation = linear
model.add(Dense(70, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1)) #output: 1개


#3. 컴파일 및 훈련


#tensorboard 
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

#log가 들어갈 폴더='graph'
#여기까지 해서 graph 폴더 생기고 자료들 들어가 있으면 텐서보드 쓸 준비 ok
#단, 로그가 많으면 겹쳐서 보일 수 있으니 그럴 땐 로그 삭제하고 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True
)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])



 
history = model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping, to_hist],
    validation_split=0.2,
    epochs=1000, batch_size=10
)


#4. 평가, 예측


loss, mae = model.evaluate(x_test, y_test, batch_size=10)
print(loss, mae)

x_predict = np.array([97, 98, 99, 100])
x_predict = x_predict.reshape(1, 4, 1)

y_predict = model.predict(x_predict)
print("예측값: ", y_predict)



# #그래프(MATPLOTLIB)
# import matplotlib.pyplot as plt

# #plt.plot에는 x, y 둘 다 넣어야 함(그래프니까)
# #loss만 넣어 놓는 건 y만 넣어 둔 것임. x=epoch임

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['mae'])
# plt.plot(history.history['val_mae'])

# #축에 명시
# plt.title('loss & mae')
# plt.ylabel('loss, mae')
# plt.xlabel('epoch')

# #그래프에 명시
# plt.legend(['train loss', 'val loss', 'train mae', 'val mae'])
# plt.show()




