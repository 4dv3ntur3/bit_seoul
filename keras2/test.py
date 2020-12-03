#2020-12-02 (18일차)
#keras DNN + GridSearchCV / RandomizedSearchCV
#그냥 튜닝해서 만든 점수 vs 최적 parameter 점수 

import datetime
import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout


from tensorflow.keras.optimizers import Adam, SGD, Nadam

from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.activations import relu, selu, elu, softmax



#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()



#전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.
# print(x_train[1])



#2. 모델

#model 
def build_model(drop=0.5, optimizer=Adam, lr=0.001, nodes=100, activation='relu'):

    inputs = Input(shape=(28*28,),name='input')

    x = Dense(nodes, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(nodes, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(nodes, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)

    outputs = Dense(10, activation=activation, name='outputs')(x) #다중 분류

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer(lr=lr), 
                  metrics=['acc'], 
                  loss='categorical_crossentropy') #one hot encoding 안 했으면 sparse_

    return model

# epochs
# batch_size
# optimizer(lr)
# dropout
# number of nodes
# activation

def create_hyperparameters():
    batches = [2000]
    optimizers= [Adam] #learning_rate 
    dropout = [0.5] #numpy.linspace 함수는 지정한 구간을 균일한 간격으로 나누는 숫자들을 반환
    epochs = [50]
    lr = [0.001]
    activation = ['relu']
    # nodes = []

    return {
           "epochs": epochs,
           "batch_size": batches, 
           "optimizer": optimizers, 
           "lr": lr,
           "drop": dropout,
           "activation": activation
           }


hyperparameters = create_hyperparameters()



###searchCV 소요 시간 측정
start = datetime.datetime.now()

#사이킷런의 GridSearchCV, RandomizedSearchCV에 keras를 넣은 것(회사가 다르다...)
#사이킷런 모델이라고 인식하게끔 해야
# -> keras를 사이킷런으로 감싸겠다
#keras -> 사이킷런에 넣을 수 있게 wrapping -> GridSearchCV / RandomizedSearchCV 사용 가능
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
search = KerasClassifier(build_fn=build_model, verbose=1)

# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model, hyperparameters, cv=3)
# search = RandomizedSearchCV(model, hyperparameters, cv=3)

search.fit(x_train, y_train)


###searchCV 소요 시간 측정
end = datetime.datetime.now()


acc = search.score(x_test, y_test)


print(search.best_params_)
print("\n최종 스코어: ", acc)
print("소요 시간: ", end-start)





'''
{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 40}
최종 스코어:  0.9707000255584717
'''

