#2020-11-18 (8일차)
#model_save(): model 구성 후 / model.fit() 후



import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #괄호 주의


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




###### 모델 저장 1)
model.save('./save/model_test01_1.h5')







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



###### 모델 저장 2) model.fit() 다음
model.save('./save/model_test01_2.h5')





#fit에 있는 네 가지
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']



#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", result[0])
print("acc: ", result[1])


#시각화
#plot에는 x, y가 들어간다 (그래야 그래프가 그려짐)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6)) 
#단위 뭔지 찾아볼 것!!!
#pyplot.figure 는 매개 변수에 주어진 속성으로 새로운 도형을 생성합니다. 
#figsize 는 도형 크기를 인치 단위로 정의합니다.


plt.subplot(2, 1, 1) #2, 1, 1 -> 두 장 중의 첫 번째의 첫 번째 (2행 1열에서 첫 번째)
# plt.plot(hist.history['loss'],) #loss값이 순서대로 감
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') #loss값이 순서대로 감, 마커는 점, 색깔은 빨간색, 라벨은 loss
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')

plt.grid() #모눈종이 배경
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')

#위에서 라벨 명시 후 위치 명시
#그림의 위치(location)는 상단: label:loss, label:val_loss 이 둘이 박스로 해서 저 위치에 나올 것
plt.legend(loc='upper right')




plt.subplot(2, 1, 2) #2, 1, 1 -> 2행 1열 중 두 번째 (두 번째 그림)
# plt.plot(hist.history['loss'],) #loss값이 순서대로 감
plt.plot(hist.history['accuracy'], marker='.', c='red') #loss값이 순서대로 감, 마커는 점, 색깔은 빨간색, 라벨은 loss
plt.plot(hist.history['val_accuracy'], marker='.', c='blue')

plt.grid() #모눈종이 배경
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

#여긴 라벨만 명시
plt.legend(['accuracy', 'val_accuracy'])

#보여 줘
plt.show()


#정답
#y_answer = np.argmax(y_answer, axis=1)

#예측값
#y_predict = model.predict(x_predict)
#y_predict = np.argmax(y_predict, axis=1)


#print("예측값: ", y_predict)
#print("정답: ", y_answer)


'''
원 결과값
acc:0.978
loss:0.0719
'''