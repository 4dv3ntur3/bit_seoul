#2020-12-02 ()
#keras(DNN) + RandomizedSearchCV <- lr + acitvation


import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Dropout


from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam
# from tensorflow.keras.activations import relu, selu, elu 


#방법1) avtivation layer를 만들어서 그 안에 넣어 준다
from tensorflow.keras.layers import Activation #acitvation layer 

#방법2) '' 안에 넣어서
from tensorflow.keras.activations import relu, selu, elu, softmax, sigmoid

#방법3) 직접 activation layer처럼 추가 
from tensorflow.keras.layers import ReLU, ELU, LeakyReLU, Softmax

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
#build_model의 parameter랑 create_hyperparameters parameter의 이름이 같아야 한다 
def build_model(drop=0.5, optimizer=Adam, lr=0.001, activation='relu'):

    inputs = Input(shape=(28*28,), name='input') 

    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x) #상위 layer의 노드 개수 감소

    x = Dense(256, name='hidden2')(x) #받은 값으로 가중치 연산 + bias, 기본값 activation=linear(받은 값 그대로. 별 의미 없음) 한 layer처럼 행동
    #그렇다면??? CNN은? LSTM은?? -> 기본 활성함수 통과 후에 또 넣어 줄지 말지는 본인의 판단
    #반드시 activation layer를 쓰라는 게 아님

    x = Activation(activation)(x)
    x = Dropout(drop)(x) #상위 layer의 노드 개수 감소 

    x = Dense(128, name='hidden3')(x)
    x = LeakyReLU(alpha=0.3)(x) #값을 전달받아서 통과시킴 (그전 layer)
    x = Dropout(drop)(x)

    outputs = Dense(10, name='outputs')(x) #다중 분류

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy') #one hot encoding 안 했으면 sparse_

    return model

#hyper parameters
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    epochs = [10, 50, 100]
    optimizers= [Adam, SGD] #learning_rate 
    dropout = np.linspace(0.1, 0.5, 5).tolist() #numpy.linspace 함수는 지정한 구간을 균일한 간격으로 나누는 숫자들을 반환합니다.
    learn_rate = [0.001, 0.001, 0.1]
    activation=['relu', 'elu', 'selu']


    return {"batch_size": batches, 
           "optimizer": optimizers, 
           "drop": dropout,
           "epochs": epochs,
           "lr": learn_rate,
           "activation": activation}

hyperparameters = create_hyperparameters()


#사이킷런의 GridSearchCV, RandomizedSearchCV에 keras를 넣은 것(회사가 다르다...)
#사이킷런 모델이라고 인식하게끔 해야
# -> keras를 사이킷런으로 감싸겠다
#keras -> 사이킷런에 넣을 수 있게 wrapping -> GridSearchCV / RandomizedSearchCV 사용 가능
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# search = GridSearchCV(model, hyperparameters, cv=3)
search = RandomizedSearchCV(model, hyperparameters, cv=3)

search.fit(x_train, y_train)
print(search.best_params_)
acc = search.score(x_test, y_test)

print("최종 스코어: ", acc)





#걸린 시간 확인

#나머지 parameter도 넣고 튜닝
#early_stopping patience
#epoch
#node수
#activation




