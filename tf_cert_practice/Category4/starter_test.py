# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# tokenizing

def solution_model():

    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    # urllib.request.urlretrieve(url, 'sarcasm.json')


    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post' #뒤를 채운다
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    # YOUR CODE HERE


    with open('sarcasm.json', 'r') as f:
        dataset = json.load(f)

    # print(dataset)

    '''
    {'article_link': 'https://www.theonion.com/odorite-introduces-new-three-tier-urinal-cake-1819592434', 
    'headline': 'odorite introduces new three-tier urinal cake', 
    'is_sarcastic': 1}
    '''

    for item in dataset:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    sentences = np.array(sentences)
    labels = np.array(labels)


    token = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    token.fit_on_texts(sentences)
    print(token.word_index) #많이 나오는 애가 앞으로 간다

    sentences = token.texts_to_sequences(sentences)

    sentences = pad_sequences(sentences, padding=padding_type, maxlen=max_length)
    # print(pad_x.shape) #(26709, 120)

    x_train = sentences[:training_size]
    x_test = sentences[training_size:]
    y_train = labels[:training_size]
    y_test = labels[training_size:]


    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(vocab_size, 256, input_length=max_length),
        tf.keras.layers.LSTM(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=256)

    print(model.evaluate(x_test, y_test, batch_size=256))

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
