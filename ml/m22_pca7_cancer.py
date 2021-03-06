#2020-11-25 (13일차)
#PCA: cancer
#column 축소해서 완성
#1. 0.95 이상
#2. 1이상 

# vs cifar10_DNN.py: loss, acc

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA #decomposition:분해


#x만 빼겠다
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape)
print(y.shape)


pca = PCA()
pca.fit(x) #fit만 됨

#누적된 합을 표시하겠다
cumsum = np.cumsum(pca.explained_variance_ratio_) #축소된 차원들의 가치의 합
d = np.argmax(cumsum >= 0.95)  + 1 #index 반환이므로 개수 확인은 +1 (0.95 이상이 되는 최소의 column 수 반환인 것 같다)
# d = np.argmax(cumsum >= 1)  + 1 #index 반환이므로 개수 확인은 +1 (0.95 이상이 되는 최소의 column 수 반환인 것 같다)

#즉, d는 우리가 필요한 n_components의 수
# print(cumsum >= 0.9) #T/F 확인
# print(d) 

pca = PCA(n_components=d) #n_components 축소할 컬럼의 수. 기존 차원보다 많으면 안 됨.
                          #MNIST의 경우 shape가 784, 인데 이걸 축소해서 넣을 수도 있겠지 다만 데이터의 손실도 있을 수 있다
                          #하지만 그걸 감수하고서라도 속도를 잡을 수 있음
x2d = pca.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)


#스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activatio n default='relu'
                                        #LSTM의 activation default='tanh'
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #2진분류: sigmoid -> output: 0 or 1 이니까 1개임 




#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=1000, batch_size=32
)



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)

print("=======cancer_dnn=======")
# model.summary()
print("column: ", d)
print("loss: ", loss)
print("acc: ", accuracy)


'''
=======cancer_dnn=======
column:  1
loss:  1.9282292127609253
acc:  0.9181286692619324
'''