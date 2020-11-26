#2020-11-26
#image data generator


#이미지 데이터의 전처리
#이미지 -> 전처리
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

np.random.seed(33) #numpy에서 난수 생성할 때 무조건 33


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


#test할 걸 반전하고 그럴 필요 없음
#기존이미지로 확인해야
test_datagen = ImageDataGenerator(rescale=1./255)

#flow(폴더 아닌 곳에서 땡겨온다. ex)데이터) 또는 flow_from_directory(폴더에서 가져온다)
#실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업



#이 디렉토리에 든 걸 가져다가 train_datagen 옵션대로 만들어라
xy_train = train_datagen.flow_from_directory(
    #경로(폴더의 위치)
    #train 바로 밑의 폴더명이 label이 된다 (0, 1)
    #파일 탐색기에서 속성으로 이미지 사이즈 확인
    './data/data1/train',
    target_size=(150, 150), #이미지 사이즈. 임의로 줘도 됨 160, 150 이렇게 
    batch_size=1, #batch_size = 200 하면 에러 남 다 돌리고 나니까 없어서... 총 160장임
    class_mode='binary' #라벨링이 여러개면 다중분류임. 즉, 라벨에 맞춰 폴더를 여러 개 생성해야 한다 
)

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150, 150), #이미지 사이즈. 임의로 줘도 됨 160, 150 이렇게 
    batch_size=1, #한 batch가 5, 150, 150, 1
    class_mode='binary'
    # , save_to_dir='./data/data1_2/train' #save 폴더
)

'''
print("===================================================")
print(type(xy_train)) 
print(xy_train[0])
# print(xy_train[0].shape)
print(xy_train[0][0])
print(type(xy_train[0][0])) #numpy.ndarray
print(xy_train[0][0]) #(5, 150, 150, 3) #맨 앞의 5=batch_size***
print(xy_train[0][1].shape) #y (5, )
#0이 많이 있다는 건 배경색
#xy_train[0][20]까지 (100/5=20)

#32개의 배열의 첫번째는 x, 두 번째는 y
# print(xy_train[1][1].shape) #y (5, )
print(len(xy_train)) #32 (160장/5 = 32)

print("===================================================")
#제일 첫 장만 보고 싶을 때
print(xy_train[0][0][0])
print(xy_train[0][1][:5])
'''


#batch_size를 full로 잡으면 0은 x, 1은 y

#이미지를 불러와서 numpy로 바꿔가지고 작업 *.npy
# np.save('./data/keras63_train_x.npy', arr=xy_train[0][0]) #batch_size = 200 해서 하면 한 번에 하나 다들어감
# np.save('./data/keras63_train_y.npy', arr=xy_train[0][1]) #batch_size = 200 해서 하면 한 번에 하나 다들어감
# np.save('./data/keras63_test_x.npy', arr=xy_test[0][0]) #batch_size = 200 해서 하면 한 번에 하나 다들어감
# np.save('./data/keras63_test_y.npy', arr=xy_test[0][1]) #batch_size = 200 해서 하면 한 번에 하나 다들어감


model = Sequential()
model.add(Conv2D(250, (2, 2), input_shape=(150, 150, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


#<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>


#train_test_split 같은 게 아니라 그냥 train이랑 test를 나눠서 generator했다
# 즉, train 안에 X, Y 다 있음
# X: 1000 * 150 * 150 * 1 (normal)
# Y: (1000, 0) (ad)   


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #fit 말고 하나 더 필요하다 (이미지 데이터셋의 경우)
history = model.fit_generator(
    xy_train, #x_train, #y_train

    steps_per_epoch = 100, #todtjdehls 이미지 중에 100개만 뽑겠다
    epochs = 20, #훈련은 스무번

    validation_data = xy_test, #validation_split
    validation_steps = 4
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']









#시각화 완성
import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot(y_loss, marker='.', c='red', label='loss')
plt.plot(y_vloss, marker=',', c='blue', label='val_loss')

plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(loc='upper right')



plt.subplot(2, 1, 2)
plt.plot(acc, marker='.', c='red')
plt.plot(val_acc, marker='.', c='blue')

plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')

plt.legend(['acc', 'val_acc'])
plt.show()