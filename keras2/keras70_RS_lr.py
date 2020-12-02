#2020-12-02 ()
#keras(DNN) + RandomizedSearchCV <- lr 


import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Dropout


from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam

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
def build_model(drop=0.5, optimizer=Adam, lr=0.001):

    inputs = Input(shape=(28*28,),name='input') 

    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)

    outputs = Dense(10, activation='softmax', name='outputs')(x) #다중 분류

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

    return {"batch_size": batches, 
           "optimizer": optimizers, 
           "drop": dropout,
           "epochs": epochs,
           "lr": learn_rate}

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


'''
{'optimizer': <class 'tensorflow.python.keras.optimizer_v2.gradient_descent.SGD'>, 'lr': 0.1, 'epochs': 100, 'drop': 0.2, 'batch_size': 40}
250/250 [==============================] - 0s 959us/step - loss: 0.0797 - acc: 0.9860
최종 스코어:  0.9860000014305115
'''


#걸린 시간 확인

#나머지 parameter도 넣고 튜닝
#early_stopping patience
#epoch
#node수
#activation


