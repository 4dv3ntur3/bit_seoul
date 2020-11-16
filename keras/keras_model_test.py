

import numpy as np


#모델을 구성하시오.
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input #LSTM도 layer

# model = Sequential()



# model.add(LSTM(30, activation='relu', input_shape=(3, 1))) # input layer 제외하고 저장 = 불가능 (error 뜸)
# model.add(Dense(50, activation='relu')) #default activation = linear
# model.add(Dense(70, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(30, activation='relu'))
# # model.add(Dense(1)) #output: 1개
# # output layer는 제외하고 저장


input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1) #상단의 input layer를 사용하겠다
dense2 = Dense(4, activation='relu')(dense1) 
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3) #선형회귀이므로 마지막 activation은 linear여야 한다

model = Model(inputs=input1, outputs=output1)



# model.summary()

#모델 저장: 경로 주의***** 
#모델의 확장자:h5
# model.save("save1.h5")
 
#\Study에 저장됨 (작업 폴더에) 즉, 비주얼스튜디오 코드 실행시 root폴더가 작업그룹이다 현재:study
#하지만 보기 편하도록 별도의 폴더를 만들겠다

#파일명에 n들어가면 \n <- 이렇게 돼서 개행 될 수도 있음 (그밖의 다른 이스케이프 문자들)
#주의!!
model.save("./save/keras26_model_2.h5") #. = root(최상위)
# model.save(".\save\keras28_2.h5")
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28_4.h5")

