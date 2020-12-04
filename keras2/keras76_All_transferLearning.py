#2020-12-04
#전이학습


#각 넷별로 하나씩은 넣어두기 
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2, ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

# t = VGG16()
# t = VGG19()
# t = Xception()
# t = ResNet101()
# t = ResNet101V2()
# t = ResNet152()
# t = ResNet152V2()
# t = ResNet50()
# t = ResNet50V2()
# t = InceptionResNetV2()
# t = InceptionV3()
# t = MobileNet()
# t = MobileNetV2()
# t = DenseNet121()
# t = DenseNet169()
# t = DenseNet201()
# t = NASNetLarge()
# t = NASNetMobile()


# model = VGG16(include_top=False, input_shape=(100, 100, 1) #매개변수 확인해 보기 #138,357,544 parameters
# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) #그럼 이렇게 줬을 때랑 파라미터 수가 똑같으면 기본값이 imagenet인가 
                                  #파라미터의 개수가 똑같다고 해서 같은 가중치일 리는 x 연산의 개수만 같을 뿐 
                                  #이미지넷인지 아닌지 어떻게 알아요
                                  #근데 이대로 갖다 쓰면 input layer가... 안 맞을... 텐데? -> input_shape 바꿔야 
                                  #include_top = False (input layer 쓰지 않겠다)
                                  #대신 input은 3차원으로 줘야 하는 듯 CNN 쓰셨나...
                                  #색깔도... 컬러여야 하는 듯... channel=3이어야 한다


t.trainable=True

t.summary() 



print("동결하기 전 훈련되는 가중치의 수: ", len(t.trainable_weights)) 
print("========================")


'''
t = VGG16() / 138,357,544 / 동결하기 전 훈련되는 가중치의 수:  32
t = VGG19() / 143,667,240 / 동결하기 전 훈련되는 가중치의 수:  38

t = Xception() 
Total params: 22,910,480
Trainable params: 22,855,952
Non-trainable params: 54,528
동결하기 전 훈련되는 가중치의 수:  156


t = ResNet101()
Total params: 44,707,176
Trainable params: 44,601,832
Non-trainable params: 105,344
__________________________________________________________________________________________________
동결하기 전 훈련되는 가중치의 수:  418


t = ResNet101V2()

t = ResNet152()
t = ResNet152V2()
t = ResNet50()
Total params: 25,636,712
Trainable params: 25,583,592
Non-trainable params: 53,120
__________________________________________________________________________________________________
동결하기 전 훈련되는 가중치의 수:  214


t = ResNet50V2()
t = InceptionResNetV2()
t = InceptionV3()
Total params: 23,851,784
Trainable params: 23,817,352
Non-trainable params: 34,432
__________________________________________________________________________________________________
동결하기 전 훈련되는 가중치의 수:  190


t = MobileNet()
t = MobileNetV2()
Total params: 3,538,984
Trainable params: 3,504,872
Non-trainable params: 34,112
__________________________________________________________________________________________________
동결하기 전 훈련되는 가중치의 수:  158



t = DenseNet121()
Total params: 8,062,504
Trainable params: 7,978,856
Non-trainable params: 83,648
__________________________________________________________________________________________________
동결하기 전 훈련되는 가중치의 수:  364


t = DenseNet169()
t = DenseNet201()
t = NASNetLarge()
t = NASNetMobile()
Total params: 5,326,716
Trainable params: 5,289,978
Non-trainable params: 36,738
__________________________________________________________________________________________________
동결하기 전 훈련되는 가중치의 수:  742






'''










'''
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
# model.add(BatchNormalization()) #가중치 수 8개 됨 -> 가중치 연산을 하는구나 (연산돼서 넘어온 값을...? )
# model.add(Dropout(0.2)) #얜 그대로 6개. 얜 가중치 연산을 안 하는구나 
model.add(Activation('relu')) #얘도 그대로 6개. 가중치 연산 x 
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

model.summary()

print("동결한 후 훈련되는 가중치의 수: ", len(model.trainable_weights)) #model.trainable=False 돼 있으면 안 나옴 #32 (LAYER 16 * (가중치 1개 + BIAS 1개) = 32)
'''