#2020-11-18 (8일차)
#fit 이후 model.save(): 모델+가중치 저장
#fit 이후 model_save_weights(): 가중치만 저장



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


#predict data, answer data
# x_predict = x_train[20:30]
# y_answer = y_train[20:30]



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

modelpath = './model/mnist-{epoch:02d}-{val_loss:.4f}.hdf5' 

#root 폴더(=Study) 밑의 model 폴더에 
#.hd5f라는 확장자 => *.h5 (모델 확장자와 거의 동일하다)
#epoch:두자릿수 정수 표기-val_loss:소수 넷째자리까지 표기 -> 이 파일명으로 생성 : 파일명만 봐도 대략 어느 지점이 좋은지 알 수 있음 

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

#val_loss가 개선이 생길 때마다 history를 저장하고 싶다!
#정의
cp = ModelCheckpoint(filepath=modelpath, 
                     monitor='val_loss', 
                     save_best_only=True, 
                     mode='auto'
) #계속 저장될 것 (계속 좋아질 테니까) 그러다가 튀는 구간 있으면 또 저장 안 되고... 다시 뚫고 내려가는 거 저장되고...
  #좋은 시점 저장됨
#model 폴더 잘 보기!
#04-어쩌고 후에 07-어쩌고가 생겼다면 5, 6은 별로였다는 뜻


#log가 들어갈 폴더='graph'
#여기까지 해서 graph 폴더 생기고 자료들 들어가 있으면 텐서보드 쓸 준비 ok
#단, 로그가 많으면 겹쳐서 보일 수 있으니 그럴 땐 로그 삭제하고 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True
)




model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy']) #"mean_squared_error" (풀네임도 가능하다)

#model.fit()은 history 반환
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
          validation_split=0.2, callbacks=[early_stopping, cp]) #tensorboard 쓰고 싶으면 여기다가 to_hist도...



#모델 + 가중치 저장
model.save('./save/model_test02_2.h5') #keras51_2_load_weights.py

#가중치만 저장
model.save_weights('./save/weight_test02.h5') #keras51_3_load_weights.py



#fit에 있는 네 가지
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']



#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", result[0])
print("acc: ", result[1])




#정답
# y_answer = np.argmax(y_answer, axis=1)

#예측값
# y_predict = model.predict(x_predict)
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