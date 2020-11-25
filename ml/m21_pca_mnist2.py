#2020-11-25 (13일차)
#PCA: mnist
#column 축소해서 완성
#1. 0.95 이상
#2. 1이상

# vs mnist_DNN.py: loss, acc

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA #decomposition:분해


#x만 빼겠다
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# print(x_train.shape) 
# print(x_test.shape) 


x = np.append(x_train, x_test, axis=0)
print(x.shape) 

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) #(70000, 28, 28)



pca = PCA()
pca.fit(x) #fit만 됨

#누적된 합을 표시하겠다
cumsum = np.cumsum(pca.explained_variance_ratio_) #축소된 차원들의 가치의 합
# d = np.argmax(cumsum >= 0.95)  + 1 #index 반환이므로 개수 확인은 +1 (0.95 이상이 되는 최소의 column 수 반환인 것 같다)
d = np.argmax(cumsum >= 1)  + 1 #index 반환이므로 개수 확인은 +1 (0.95 이상이 되는 최소의 column 수 반환인 것 같다)

#즉, d는 우리가 필요한 n_components의 수
# print(cumsum >= 0.9) #T/F 확인
# print(d) 

pca = PCA(n_components=d) #n_components 축소할 컬럼의 수. 기존 차원보다 많으면 안 됨.
                          #MNIST의 경우 shape가 784, 인데 이걸 축소해서 넣을 수도 있겠지 다만 데이터의 손실도 있을 수 있다
                          #하지만 그걸 감수하고서라도 속도를 잡을 수 있음
x2d = pca.fit_transform(x)


x_train = x2d[:60000]
x_test = x2d[60000:]


x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_predict = x_train[20:30]
y_answer = y_train[20:30]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(x_train.shape[1],))) 
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
# model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))




#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', patience=20, mode='auto')


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
callbacks=[es])


#4. 훈련 및 평가
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", loss)
print("acc: ", accuracy)


'''
0.95
loss:  0.3113694489002228
acc:  0.9757000207901001

1
loss:  0.34476107358932495
acc:  0.9729999899864197
'''

#줄여서 했는데도 전체 column을 다 넣은 것과 동일하게 나온다면 성능(속도)를 ㅟ해서는 
#pca 하는 게 더 나음!