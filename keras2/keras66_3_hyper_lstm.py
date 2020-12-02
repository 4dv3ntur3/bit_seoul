#2020-12-02 ()
#keras LSTM + RandomizedSearchCV
#그냥 튜닝해서 만든 점수 vs 최적 parameter 점수 


import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Dropout, Flatten, LSTM



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
def build_model(drop=0.5, optimizer='adam'):

    # inputs = Input(shape=(28*28,),name='input') 

    # x = Dense(512, activation='relu', name='hidden1')(inputs)
    # x = Dropout(drop)(x)
    # x = Dense(256, activation='relu', name='hidden2')(x)
    # x = Dropout(drop)(x)
    # x = Dense(128, activation='relu', name='hidden3')(x)
    # x = Dropout(drop)(x)

    # outputs = Dense(10, activation='softmax', name='outputs')(x) #다중 분류

    # model = Model(inputs=inputs, outputs=outputs)
    model = Sequential()
    model.add(LSTM(1000, activation='relu', input_shape=(28, 28))) #꼭 28, 28, 1일 필요는 없음 #뭔가 시계열 같은 데이터라고 판단이 되면 몇 개씩 자를지 생각할 수도 있음
    model.add(Dense(500, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid(여자/남자, dead/alive)
                                                #즉 softmax를 사용하려면 OneHotEncoding 해야



    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy') #one hot encoding 안 했으면 sparse_

    return model

#hyper parameters
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers= ['rmsprop', 'adam', 'adadelta'] #learning_rate 
    dropout = np.linspace(0.1, 0.5, 5).tolist() #numpy.linspace 함수는 지정한 구간을 균일한 간격으로 나누는 숫자들을 반환합니다.
    
    #추가할 수 있는 거 
    #early_stopping
    #epoch
    #pipeline 엮기 
    #activation...


    return {"batch_size": batches, 
           "optimizer": optimizers, 
           "drop": dropout}

hyperparameters = create_hyperparameters()


#사이킷런의 GridSearchCV, RandomizedSearchCV에 keras를 넣은 것(회사가 다르다...)
#사이킷런 모델이라고 인식하게끔 해야
# -> keras를 사이킷런으로 감싸겠다
#keras -> 사이킷런에 넣을 수 있게 wrapping -> GridSearchCV / RandomizedSearchCV 사용 가능
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



#걸린 시간 확인

#나머지 parameter도 넣고 튜닝
#early_stopping patience
#epoch

#딥러닝
# y = wx + b
#가장 기본이 되는 게 경사 하강법 
