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


#이미지에 대한 생성 옵션 정하기: 튜닝 가능
train_datagen = ImageDataGenerator(rescale=1./255, #정규화
                                    horizontal_flip=True, #수평
                                    vertical_flip=True, #수직
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    rotation_range=5,
                                    zoom_range=1.2,
                                    shear_range=0.7,
                                    fill_mode='nearest') #빈자리를 주변과 비슷하게 채워준다. 0 하면 padding


test_datagen = ImageDataGenerator(rescale=1./255)


xy_train = train_datagen.flow_from_directory(
    #경로(폴더의 위치)
    #train 바로 밑의 폴더명이 label이 된다 (0, 1)
    #파일 탐색기에서 속성으로 이미지 사이즈 확인
    './data/data2/train',
    target_size=(100, 100), #이미지 사이즈. 임의로 줘도 됨 160, 150 이렇게 
    batch_size=100, #16으로 주면 앞에가 10개겠지 160장/16 = 10
    class_mode='binary' #라벨링이 여러개면 다중분류임. 즉, 라벨에 맞춰 폴더를 여러 개 생성해야 한다 
)

xy_test = test_datagen.flow_from_directory(
    './data/data2/test',
    target_size=(100, 100),
    batch_size = 100,
    class_mode='binary'
)


np.save('./data/keras64_train_x.npy', arr=xy_train[0][0]) #batch_size = 200 해서 하면 한 번에 하나 다들어감
np.save('./data/keras64_train_y.npy', arr=xy_train[0][1]) #batch_size = 200 해서 하면 한 번에 하나 다들어감
np.save('./data/keras64_test_x.npy', arr=xy_test[0][0]) #batch_size = 200 해서 하면 한 번에 하나 다들어감
np.save('./data/keras64_test_y.npy', arr=xy_test[0][1]) #batch_size = 200 해서 하면 한 번에 하나 다들어감


model = Sequential()
model.add(Conv2D(1000, (10, 10), input_shape=(100, 100, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #fit 말고 하나 더 필요하다 (이미지 데이터셋의 경우)

history = model.fit_generator(
    xy_train, #x_train, #y_train

    steps_per_epoch = 5, #todtjdehls 이미지 중에 100개만 뽑겠다
    epochs = 100, #훈련은 스무번

    validation_data = xy_test, #validation_split
    validation_steps = 4
)



