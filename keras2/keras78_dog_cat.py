#2020-12-04

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import numpy as np

#전처리
from tensorflow.keras.preprocessing.image import load_img

#shape 맞춰 주는 parameter
#224에 맞춰 줬으니까 VGG16 input_top=False 할 필요 없음 (날릴 필요 없음)
img_dog = load_img('./data/dog_cat/개.jpg', target_size=(224, 224))
img_cat = load_img('./data/dog_cat/고양이.jpg', target_size=(224, 224))
img_suit = load_img('./data/dog_cat/슈트.jpg', target_size=(224, 224))
img_lion = load_img('./data/dog_cat/라이언.jpg', target_size=(224, 224))



#이미지 확인 
# plt.imshow(img_dog)
# plt.show()

#imagenet: 수천~수만 개의 각종 개체 사진이 들어가 있다 & 종류

#이미지를 데이터로 변환해 줘야 한다 
from tensorflow.keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_lion = img_to_array(img_lion)


# print(arr_dog)

# type:  <class 'numpy.ndarray'>
# shape:  (918, 919, 3)

print("type: ", type(arr_dog))
print("shape: ", arr_dog.shape)

# 이미지 구성: RGB 형태인데, keras에서 가져올 때는 BGR 형태로 가져오게 됨
# 그러므로 keras에서 사용하려면 RGB -> BGR 형태로 바꿔야 됨 reshape할 수도 있겠지만 tool 있다

from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_lion = preprocess_input(arr_lion)

#shape만 봐선 알 수 없음... 내용 보면 달라져 있다. 아무튼 맨 끝의 값이 바뀌었다 
#그대로 하면 색이 달라짐 
print(arr_dog.shape)    #(918, 919, 3)
print(arr_cat.shape)    #(1200, 1200, 3)

# print(arr_dog)

arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_lion])
print(arr_input.shape)  #(2, 918, 918, 3)



#2. 모델 구성
model = VGG16()
probs = model.predict(arr_input)

print(probs)
print('probs.shape : ', probs.shape) #probs.shape :  (2, 1000) 
#VGG16은 가중치 imagenet 사용한다... imagenet에 들어가 있는 1000개의 class

#결과 확인을 위해 디코딩
from tensorflow.keras.applications.vgg16 import decode_predictions

results = decode_predictions(probs)


#1000개 중에 가장 확률이 높은 놈을 고른다 

print("=========================")
print("results: ", results[0])
print("=========================")
print("results: ", results[1])
print("=========================")
print("results: ", results[2])
print("=========================")
print("results: ", results[3])


'''
results:  [('n02113624', 'toy_poodle', 0.7509314), ('n02113712', 'miniature_poodle', 0.23522107), ('n02113799', 'standard_poodle', 0.0043491586), ('n02102318', 'cocker_spaniel', 0.0027509595), ('n04399382', 'teddy', 0.0019554272)]
=========================
results:  [('n02123045', 'tabby', 0.33127898), ('n02124075', 'Egyptian_cat', 0.1514997), ('n02123159', 'tiger_cat', 0.046995547), ('n04493381', 'tub', 0.024293292), ('n02808440', 'bathtub', 0.021043729)]
=========================
results:  [('n04350905', 'suit', 0.786063), ('n02883205', 'bow_tie', 0.114983425), ('n04591157', 'Windsor_tie', 0.043323025), ('n02963159', 'cardigan', 0.030769063), ('n03404251', 'fur_coat', 0.00461211)]
=========================
results:  [('n04548280', 'wall_clock', 0.25236833), ('n02708093', 'analog_clock', 0.078438014), ('n03532672', 'hook', 0.061394814), ('n03291819', 'envelope', 0.04905947), ('n04127249', 'safety_pin', 0.037014622)]
'''