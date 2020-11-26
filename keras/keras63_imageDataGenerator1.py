#2020-11-26 
#image Data Generator
#이미지 규격이 일정해야 한다


#이미지 데이터의 전처리
#이미지 -> 전처리
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

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
    batch_size=5, #16으로 주면 앞에가 10개겠지 160장/16 = 10
    class_mode='binary' #라벨링이 여러개면 다중분류임. 즉, 라벨에 맞춰 폴더를 여러 개 생성해야 한다 
)

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150, 150), #이미지 사이즈. 임의로 줘도 됨 160, 150 이렇게 
    batch_size=5, #한 batch가 5, 150, 150, 1
    class_mode='binary'
    # , save_to_dir='./data/data1_2/train' #save 폴더
)


#train_test_split 같은 게 아니라 그냥 train이랑 test를 나눠서 generator했다
#즉, train 안에 X, Y 다 있음
#X: 1000 * 150 * 150 * 1 (normal)
#Y: (1000, 0) (ad)   


#fit 말고 하나 더 필요하다 (이미지 데이터셋의 경우)
model.fit_generator(
    xy_train, #x_train, #y_train

    #이 parameter의 비밀은 확인하기
    steps_per_epoch = 100, #todtjdehls 이미지 중에 100개만 뽑겠다
    epochs = 20, #훈련은 스무번

    validation_data = xy_test, #validation_split
    validation_steps = 4
)