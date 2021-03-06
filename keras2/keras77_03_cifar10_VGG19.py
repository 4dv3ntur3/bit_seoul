#2020-12-04
#cifar10 + VGG19 


from tensorflow.keras.applications import VGG19
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
t = VGG19(weights='imagenet', include_top=False, input_shape=(x_train.shape[1], x_train.shape[2], 3))
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

print("==========cifar10_VGG19==========")
model.summary()
print("loss: ", loss)
print("acc: ", accuracy)




'''
==========cifar10_VGG19==========
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg19 (Functional)           (None, 1, 1, 512)         20024384
_________________________________________________________________
flatten (Flatten)            (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 256)               131328
_________________________________________________________________
batch_normalization (BatchNo (None, 256)               1024
_________________________________________________________________
dropout (Dropout)            (None, 256)               0
_________________________________________________________________
activation (Activation)      (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570
=================================================================
Total params: 20,225,098
Trainable params: 200,202
Non-trainable params: 20,024,896
_________________________________________________________________
loss:  2.10265851020813
acc:  0.5878999829292297
'''