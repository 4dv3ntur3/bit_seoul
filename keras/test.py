
#1. 데이터
import numpy as np

# input 1개
x1=np.array([range(1,101), range(711, 811), range(100)])

# output 3개
y1=np.array([range(101,201), range(311, 411), range(100)])
y2=np.array([range(501,601), range(431,531), range(100,200)])
y3=np.array([range(501,601), range(431,531), range(100,200)])

x1=np.transpose(x1)

y1=np.transpose(y1)
y2=np.transpose(y2)
y3=np.transpose(y3)

print(x1.shape) 
print(y1.shape)
print(y2.shape)
print(y3.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test = train_test_split(
    x1, shuffle=True, train_size=0.6
)

from sklearn.model_selection import train_test_split
y1_train, y1_test, y2_train,y2_test, y3_train, y3_test = train_test_split(
    y1,y2,y3, shuffle=True, train_size=0.6
)

# from sklearn.model_selection import train_test_split
# y3_train, y3_test = train_test_split(
#     y3, shuffle=True, train_size=0.2
# )


#2. 함수형 모델 2개 구성

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape=(3,))
dense1_1 = Dense(10, activation='relu', name='king1')(input1)
dense1_2 = Dense(7, activation='relu', name='king2')(dense1_1)
dense1_3 = Dense(5, activation='relu', name='king3')(dense1_2)
output_1 = Dense(3, activation='linear', name='king4')(dense1_3)
model1 = Model(inputs=input1, outputs=output_1)

# model1.summary()

# 모델2
# input2 = Input(shape=(3,))
# dense2_1 = Dense(15, activation='relu', name='qeen1')(input2)
# dense2_2 = Dense(11, activation='relu', name='qeen2')(dense2_1)
# output2 = Dense(3, activation='linear', name='qeen3')(dense2_2) #activation='linear'인 상태

# model2 = Model(inputs=input2, outputs=output2)

# model2.summary()

#모델 병합, concatenate
# from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import Concatenate, concatenate
# from keras.layers import Concatenate, concatenate

#대문자는 클래스 
# merge1 = concatenate([output1, output2]) #2개 이상이라 list로 묶습니다
# merge1 = Concatenate()([output1, output2]) 대문자로 쓰기 위한 방법 1
# merge1 = Concatenate(axis=1)([output1, output2]) 대문자로 쓰기 위한 방법 2

# middle1 = Dense(30)(merge1)
# middle2 = Dense(7)(middle1)
# middle3 = Dense(11)(middle2)

#이름 이것도 가능 (다만, 가독성 위해 이름을 middle 1, 2, 3)
# middle1 = Dense(30)(merge1)
# middle1 = Dense(7)(middle1)
# middle1 = Dense(11)(middle1)

################# output 모델 구성 (분기)
output1 = Dense(30)(output_1)
output1_1 = Dense(7)(output1)
output1_2 = Dense(3)(output1_1)

output2 = Dense(15)(output_1)
output2_1 = Dense(14)(output2)
output2_3 = Dense(11)(output2_1)
output2_4 = Dense(3)(output2_3)

output3 = Dense(20)(output_1)
output3_1 = Dense(14)(output3)
output3_3 = Dense(11)(output3_1)
output3_4 = Dense(3)(output3_3)

#2 모델 정의
model = Model(inputs = input1, 
              outputs = [output1_2, output2_4,output3_4])

model.summary()

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=8,
           validation_split=0.25, verbose=1)

result = model.evaluate(x1_test,[y1_test,y2_test, y3_test], batch_size=8)

print("result: ", result)