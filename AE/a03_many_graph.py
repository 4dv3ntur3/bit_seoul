#2020-12-16
#Auto Encoder + MNIST
#함수 정의해서 모델 구성 
#AE : 채색, 잡티 제거에 주로 쓰임 


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

# 함수를 사용하는 이유: 재사용
def autoencoder(hidden_layer_size):

    model = Sequential()
    model.add(Dense(units=hidden_layer_size, 
                    input_shape=(784,),
                    activation='relu'))

    model.add(Dense(units=784, activation='sigmoid'))
    return model

#PCA에서 MNIST 할 때 0.95이상일 때 column 154였음 
model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08 = autoencoder(hidden_layer_size=8)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)


model_01.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_01.fit(x_train, x_train, epochs=10, batch_size=256,
                validation_split=0.2)


model_02.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_02.fit(x_train, x_train, epochs=10, batch_size=256,
                validation_split=0.2)


model_04.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_04.fit(x_train, x_train, epochs=10, batch_size=256,
                validation_split=0.2)


model_08.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_08.fit(x_train, x_train, epochs=10, batch_size=256,
                validation_split=0.2)


model_16.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_16.fit(x_train, x_train, epochs=10, batch_size=256,
                validation_split=0.2)


model_32.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_32.fit(x_train, x_train, epochs=10, batch_size=256,
                validation_split=0.2)


output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_04 = model_04.predict(x_test)
output_08 = model_08.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)



# 그림 확인
import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(7, 5, figsize=(15, 15))

random_imgs = random.sample(range(output_01.shape[0]), 5)
outputs = [x_test, output_01, output_02, output_04, output_08, output_16, output_32]
# 노드를 늘릴수록 특성을 잘 잡는다. 적을수록 흐릿하게 나옴 

for row_num, row in enumerate(axes): # row_num의 index 번호, row에 해당 내용 
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28), cmap='gray')

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()