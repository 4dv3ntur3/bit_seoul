#*.npy 불러와
#.fit으로 코딩


#2020-11-26 
#image data generator
#남자 / 여자
#image *.npy로 저장 
#fit_generator로 코딩
#train_test 알아서 분리

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
import numpy as np


x_train = np.load('./data/keras64_train_x.npy') #batch_size = 200 해서 하면 한 번에 하나 다들어감
y_train = np.load('./data/keras64_train_y.npy')
x_test = np.load('./data/keras64_test_x.npy') #batch_size = 200 해서 하면 한 번에 하나 다들어감
y_test = np.load('./data/keras64_test_y.npy') #batch_size = 200 해서 하면 한 번에 하나 다들어감



from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


model = Sequential()
model.add(Conv2D(1000, (10, 10), input_shape=(100, 100, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #fit 말고 하나 더 필요하다 (이미지 데이터셋의 경우)


model.fit(x_train, y_train, batch_size=30, epochs=100)

results = model.evaluate(x_test, y_test, batch_size=50)
print("loss, acc: ", results)



