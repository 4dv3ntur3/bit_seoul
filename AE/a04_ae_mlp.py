#2020-12-16
#DAE
#a02_ae.py 복사 + 딥하게 구성 (layer 추가)
#어떻게 보면 레이어를 통과할 때마다 값이 변하니까 오히려 hidden layer 추가 안 하는 게 값이 더 잘 나올 수도 있다 

import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data() #(60000, 28, 28) (60000,)
(x_train, _), (x_test, _) = mnist.load_data() # y값 안 가져옴 (이미지는 가져오되 이미지에 대한 라벨이 없는 상태)


# 두 방법 모두 상관없음 
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784)/255.

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#함수형 sequential
#CNN output: filter
#LSTM layer output naming:


def autoencoder(hidden_layer_size):

    model = Sequential()
    model.add(Dense(units=hidden_layer_size, 
                    input_shape=(784,),
                    activation='relu'))

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(512, activation='relu'))

    model.add(Dense(units=784, activation='sigmoid'))
    return model

#PCA에서 MNIST 할 때 0.95이상일 때 column 154였음 
model = autoencoder(hidden_layer_size=154) #784 -> 154 -> 784

#비교하기
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.compile(optimizer='adam', loss='mse', metrics=['acc']) # acc가 0.01 ㅎㅎ... 믿을 놈은 loss뿐 


#x로 x 확인
model.fit(x_train, x_train, epochs=10, batch_size=256,
                validation_split=0.2)

output = model.predict(x_test)


#x_test를 넣었을 때 x_test가 정상적으로 나오면 잘된 것
#차원축소 후 증폭하는 개념 
decoded_img = model.predict(x_test)



# 그림 확인
import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7)) # \로 이어 주거나 걍 한 줄에 쓰거나 

# 이미지 다섯 개 무작위 선택
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')

    if i == 0:
        ax.set_ylabel('INPUT', size=20) #글씨 사이즈 

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# AE가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')

    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()