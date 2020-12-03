#2020-12-02 (18일차)
#keras CNN + RandomizedSearchCV
#그냥 튜닝해서 만든 점수 vs 최적 parameter 점수 

import datetime
import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Dropout, Flatten

from tensorflow.keras.optimizers import Adam, SGD, Nadam

from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.activations import relu, selu, elu, softmax

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()



#전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
# print(x_train[1])


#왜 dropout을 ndarray로 주면 안 돌아가는지... type 찍어 보려고 확인
# dropout = np.linspace(0.1, 0.5, 5)
# print(dropout)

# dropout = np.linspace(0.1, 0.5, 5).tolist()
# print(dropout)



#2. 모델

#model 
def build_model(drop=0.5, optimizer=Adam, lr=0.001, activation='relu', pool=2, strides=2, kernel=2):
    
    model = Sequential()
    model.add(Conv2D(30, (kernel, kernel), padding='same', input_shape=(28, 28, 1))) #padding 주의!
    model.add(Conv2D(50, (kernel, kernel), padding='valid'))
    model.add(Conv2D(120, (kernel, kernel))) #padding default=valid
    model.add(Conv2D(200, (kernel, kernel), strides=strides))
    model.add(Conv2D(30, (kernel, kernel)))
    model.add(MaxPooling2D(pool_size=pool)) #pool_size default=2
    model.add(Flatten()) 
    model.add(Dense(10, activation=activation))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy') #one hot encoding 안 했으면 sparse_

    return model

#hyper parameter
# epochs
# batch_size
# optimizer(lr)
# dropout
# activation
# pool_size
# strides
# kernel_size

def create_hyperparameters():
    batches = [10, 50, 100]
    optimizers= [Adam, Nadam, SGD] #learning_rate 
    dropout = np.linspace(0.1, 0.5, 5).tolist() #numpy.linspace 함수는 지정한 구간을 균일한 간격으로 나누는 숫자들을 반환
    epochs = [10, 50, 100]
    lr = [0.001, 0.002, 0.003]
    activation = ['relu', 'selu', 'elu']
    pool_size = [2, 3]
    strides = [2, 3]
    kernel_size = [2, 3, 4]

    return {
           "epochs": epochs,
           "batch_size": batches, 
           "optimizer": optimizers, 
           "lr": lr,
           "drop": dropout,
           "activation": activation,
           "pool": pool_size,
           "strides": strides,
           "kernel": kernel_size
           }

hyperparameters = create_hyperparameters()

###searchCV 소요 시간 측정
start = datetime.datetime.now()

#사이킷런의 GridSearchCV, RandomizedSearchCV에 keras를 넣은 것(회사가 다르다...)
#사이킷런 모델이라고 인식하게끔 해야
# -> keras를 사이킷런으로 감싸겠다
#keras -> 사이킷런에 넣을 수 있게 wrapping -> GridSearchCV / RandomizedSearchCV 사용 가능
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=3)

search.fit(x_train, y_train)




###searchCV 소요 시간 측정
end = datetime.datetime.now()


acc = search.score(x_test, y_test)


print(search.best_params_)
print("\n최종 스코어: ", acc)
print("소요 시간: ", end-start)

'''
{'optimizer': 'adam', 'drop': 0.30000000000000004, 'batch_size': 50}
최종 스코어:  0.9695000052452087
'''

