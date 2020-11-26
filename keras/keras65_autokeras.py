#2020-11-26 
#autokeras

#pip install autokeras
#pip instal git+https://github.com/keras-team/keras-tuner.git@1.0.2rc4

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak



x_train = np.load('./data/keras64_train_x.npy') #batch_size = 200 해서 하면 한 번에 하나 다들어감
y_train = np.load('./data/keras64_train_y.npy')
x_test = np.load('./data/keras64_test_x.npy') #batch_size = 200 해서 하면 한 번에 하나 다들어감
y_test = np.load('./data/keras64_test_y.npy') #batch_size = 200 해서 하면 한 번에 하나 다들어감


print(x_train.shape)
print(x_test.shape)
print(y_train[:3])

#initialize the image classifier
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=10 #default=1
)

#feed the image calssifier with training data
clf.fit(x_train, y_train, epochs=20) #epoch 바꿔 보기


#predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)

#evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
#clf.summary() 안 먹힌다 



#튜닝이 거의 필요 없다


'''
[1.5148749351501465, 0.5799999833106995]
'''