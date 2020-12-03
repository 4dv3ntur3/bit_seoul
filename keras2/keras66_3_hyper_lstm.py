#2020-12-02 (18일차)
#keras LSTM + RandomizedSearchCV
#그냥 튜닝해서 만든 점수 vs 최적 parameter 점수 


import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Dropout, Flatten, LSTM

from tensorflow.keras.optimizers import Adam, SGD, Nadam

from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.activations import relu, selu, elu, softmax

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()



#전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



x_train = x_train.reshape(60000, 28, 28).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28).astype('float32')/255.
# print(x_train[1])


# dropout = np.linspace(0.1, 0.5, 5)
# print(dropout)

# dropout = np.linspace(0.1, 0.5, 5).tolist()
# print(dropout)



#2. 모델

#model 
def build_model(drop=0.5, optimizer=Adam, lr=0.001, activation='relu'):

    model = Sequential()
    model.add(LSTM(1000, activation=activation, input_shape=(28, 28))) #꼭 28, 28, 1일 필요는 없음 #뭔가 시계열 같은 데이터라고 판단이 되면 몇 개씩 자를지 생각할 수도 있음
    model.add(Dense(500, activation=activation))
    model.add(Dense(300, activation=activation))
    model.add(Dense(200, activation=activation))
    model.add(Dense(50, activation=activation))
    model.add(Dense(10, activation=softmax))

    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy') #one hot encoding 안 했으면 sparse_

    return model

#hyper parameters
# epochs
# batch_size
# optimizer(lr)
# dropout
# number of nodes
# activation
# validation_split 

def create_hyperparameters():
    batches = [100, 500, 1000]
    optimizers= [Adam, Nadam, SGD] #learning_rate 
    dropout = np.linspace(0.1, 0.5, 5).tolist() #numpy.linspace 함수는 지정한 구간을 균일한 간격으로 나누는 숫자들을 반환
    epochs = [10, 50, 100]
    lr = [0.001, 0.002, 0.003]
    activation = ['relu', 'selu', 'elu']
    val = [0.2, 0.3]

    return {
           "epochs": epochs,
           "batch_size": batches, 
           "optimizer": optimizers, 
           "lr": lr,
           "drop": dropout,
           "activation": activation,
           "validation_split": val
           }

hyperparameters = create_hyperparameters()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=3)

search.fit(x_train, y_train)

acc = search.score(x_test, y_test)

print(search.best_params_)
print("최종 스코어: ", acc)



'''


'''
