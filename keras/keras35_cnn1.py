#2020-11-16 (6일차)
#CNN: 조각조각 잘라서 그 중에서 특성값을 빼낸다 
#Conv2D
#flatten

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten #그림=2차원 1D, 3D도 있음

model = Sequential()


#output node수는 바꿀 수 있으므로 설정하기...
# model.add(Conv2D(10, (2, 2), input_shape=(5,5,1))) #output: 4, 4, (5,5 + kernel 2,2) 10(output node) 실질적으로 용량 늘어남 10 - n + 1
                    #kernel_size (이렇게 잘라서 )
#이미지 자른 게 5*5, 흑백=1, color=3 #(2,2) -> 2*2씩 자르겠다

model.add(Conv2D(10, (2, 2), input_shape=(10, 10, 1))) #ouput: 9 9 10
model.add(Conv2D(5, (2, 2), padding='same')) #input_shape 안 써 줘도 됨 (위에서 받음) output: 3, 3, 5(output node)  output:8 8 5 + padding -> 9, 9, 5
model.add(Conv2D(3, (3, 3), padding='valid')) #output: 7, 7, 3 (위층 padding)
model.add(Conv2D(7, (2, 2))) # 6, 6, 7



#data의 수치가 큰 게 feature값이 더 높다
#Maxpooling2D: 가장 특성치가 높은 것만 남긴다(특성 추출) (데이터 양 확 줄여 줌)
model.add(MaxPooling2D()) #default pool size=(2,2)  (3, 3, 7) #반으로 줄어든다

#문제: Dense에 전달할 수 없다 (여전히 출력이 3차원이라서)
#reshape는 복잡함
#받은 데이터를 일렬로 쫙 편다 (나열한다)
model.add(Flatten())    # 3*3*7  = 63
model.add(Dense(1)) #최종 output



model.summary()
#Conv2D parameter numbers
#number_parameters = out_channels * (in_channels * kernel_h * kernel_w + 1)  1 for bias
#maxPooling2D랑 Flatten은 연산하지 않음



#convolution에서 출력되는 output 차원은 input과 동일하다****

#Conv2D(filters, kernel_size, strides, input_shape)
# 10 : filters(아웃풋, 다음 layer에 던져 줄 node의 개수) Integer, the dimensionality of the output space, 튜닝 가능
# (2,2): kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
# strides(보폭): specifying the strides of the convolution along the height and width. default=1 (kernel_size를 몇 칸씩 건너뛸지)
# padding: "valid" / "same" (적용x / 적용o)
# 입력: (batch_size, rows, cols, channels) 4차원 (channel***) batch_size=이미지 수 (행 전체에서 얼마큼 잘라서 작업하는지= 즉 행)
# channels: 이미지 픽셀 하나하나는 실수, 
# 컬러 사진은 천연색을 표현하기 위해서, 각 픽셀을 RGB 3개의 실수로 표현한 3차원 데이터입니다.
# 컬러 이미지는 3개의 채널로 구성됩니다. 반면에 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성됩니다. 
# 높이가 39 픽셀이고 폭이 31 픽셀인 컬러 사진 데이터의 shape은 (39, 31, 3)3으로 표현합니다. 
# 반면에 높이가 39픽셀이고 폭이 31픽셀인 흑백 사진 데이터의 shape은 (39, 31, 1)입니다.
# input_shape: (rows, cols, channels) 3차원

#LSTM parameters
# LSTM(200, activation='relu', input_shape=(3, 1), return_sequences=True)
# units: 노드 수
# 입력: (batch_size, timesteps, feature) 3차원 (가로, 세로, 특성)
# input_shape: (timesteps, feature) 2차원 #며칠분씩 잘랐는지(시간 간격의 규칙) 즉 열, 몇개씩 자르는지 
# return_sequences: True / False










