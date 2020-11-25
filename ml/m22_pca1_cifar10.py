#2020-11-25 (13일차)
#PCA: cifar-10
#column 축소해서 완성
#1. 0.95 이상
#2. 1이상 = 32*32*3 =3072랑 같다 (안 변함)

# vs cifar10_DNN.py: loss, acc

import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA #decomposition:분해


#x만 빼겠다
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) #50000, 32, 32, 3
print(x_test.shape) 

x_train_shape = x_train.shape[0]
x_test_shape = x_test.shape[0]


#PCA도 스케일링이 되긴 하지만 신뢰할 수 없다면 하는 게 낫다
#n_component에 비율 써도 괜찮다


x = np.append(x_train, x_test, axis=0)
print(x.shape) 


x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])



pca = PCA()
pca.fit(x) #fit만 됨

#누적된 합을 표시하겠다
cumsum = np.cumsum(pca.explained_variance_ratio_) #축소된 차원들의 가치의 합
d = np.argmax(cumsum >= 0.95)  + 1 #index 반환이므로 개수 확인은 +1 (0.95 이상이 되는 최소의 column 수 반환인 것 같다)
# d = np.argmax(cumsum >= 1)  + 1 #index 반환이므로 개수 확인은 +1 (0.95 이상이 되는 최소의 column 수 반환인 것 같다)

#즉, d는 우리가 필요한 n_components의 수
# print(cumsum >= 0.9) #T/F 확인


pca = PCA(n_components=d) #n_components 축소할 컬럼의 수. 기존 차원보다 많으면 안 됨.
                          #MNIST의 경우 shape가 784, 인데 이걸 축소해서 넣을 수도 있겠지 다만 데이터의 손실도 있을 수 있다
                          #하지만 그걸 감수하고서라도 속도를 잡을 수 있음
x2d = pca.fit_transform(x)


x_train = x2d[:x_train_shape]
x_test = x2d[x_train_shape:]


x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.



#스케일링은 여기서


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# x_predict = x_train[20:30]
# y_answer = y_train[20:30]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(3072,))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activation default='relu'
                                        #LSTM의 activation default='tanh'
                                        
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(3000, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid(여자/남자, dead/alive)
                                            #즉 softmax를 사용하려면 OneHotEncoding 해야

print("모델까지 됨")

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy']) #"mean_squared_error" (풀네임도 가능하다)

model.fit(x_train, 
          y_train, 
          epochs=100, 
          batch_size=512, 
          validation_split=0.2, 
          callbacks=[early_stopping])



#4. 평가, 예측
#fit에서 쓴 이름과 맞춰 주기 


loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)

print("======cifar10_DNN=======")
# model.summary()
print(d) 
print("loss: ", loss)
print("acc: ", accuracy)


# 0.95 
# 217
# loss:  5.1018571853637695
# acc:  0.42989999055862427


