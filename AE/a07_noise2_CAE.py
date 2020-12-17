#2020-12-16
#기미 주근깨를 제거하려면 기미 주근깨가 먼저 있어야지... -> noise 추가

import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data() #(60000, 28, 28) (60000,)
(x_train, _), (x_test, _) = mnist.load_data() # y값 안 가져옴 (이미지는 가져오되 이미지에 대한 라벨이 없는 상태)


# 두 방법 모두 상관없음 
x_train_input = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_train_output = x_train.reshape(60000, 784).astype('float32')/255


x_test_input = x_test.reshape(10000, 28, 28, 1)/255.
x_test_output = x_test.reshape(10000, 784)/255.

# 전처리했으므로 0~1사이의 데이터에 random.normal로 뽑은 숫자 더해준다 -> 점이 찍힘 
# 그런데 1.0 넘어간 애들이 생기는데, 걔네를 출력하면 제대로 나올까? imshow는 0~1사이만 나오는데  

# print(x_train[0])
# print(x_test[0])


x_train_input_noised = x_train_input + np.random.normal(0, 0.1, size=x_train_input.shape) #0에서 0.1 사이를 x_train의 값에 더해준다 
x_test_input_noised = x_test_input + np.random.normal(0, 0.1, size=x_test_input.shape)

x_train_input_noised = np.clip(x_train_input_noised, a_min=0, a_max=1)
x_test_input_noised = np.clip(x_test_input_noised, a_min=0, a_max=1) 


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten


def autoencoder(hidden_layer_size):
    
    model = Sequential()

    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2),input_shape=(28, 28, 1), activation='relu', padding='same'))

    model.add(Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same'))

    model.add(Flatten())

    model.add(Dense(units=784, activation='sigmoid'))

    return model

#PCA에서 MNIST 할 때 0.95이상일 때 column 154였음 
model = autoencoder(hidden_layer_size=154) #784 -> 154 -> 784

#비교하기
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.compile(optimizer='adam', loss='mse', metrics=['acc']) # acc가 0.01 ㅎㅎ... 믿을 놈은 loss뿐 


model.fit(x_train_input_noised, x_train_output, epochs=10, batch_size=256,
                validation_split=0.2)

output = model.predict(x_test_input_noised)


# 그림 확인
import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7)) # \로 이어 주거나 걍 한 줄에 쓰거나 

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

# noise 추가한 이미지를 그 다음에 
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_input_noised[random_images[i]].reshape(28, 28), cmap='gray')

    if i == 0:
        ax.set_ylabel('INPUT_NOISED', size=20) #글씨 사이즈 

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# AE가 출력한 이미지를 맨 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')

    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


# 추가한 noise가 없어져서 나오는데, 수치가 높지 않아서 제거된 것임(아니면 너무 높든가)
