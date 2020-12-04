#2020-12-04
#cifar10 + Xception


from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3).astype('float32')/255.



#2. 모델
t = Xception(weights='imagenet', include_top=False, input_shape=(x_train.shape[1], x_train.shape[2], 3))
t.trainable=False #학습시키지 않겠다 이미지넷 가져다가 그대로 쓰겠다 
# model.trainable=True

model = Sequential()
model.add(t)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=512)


#4. 평가
loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)

print("==========cifar10_Xception==========")
model.summary()
print("loss: ", loss)
print("acc: ", accuracy)

'''
ValueError: Input size must be at least 71x71; got `input_shape=(32, 32, 3)`
'''