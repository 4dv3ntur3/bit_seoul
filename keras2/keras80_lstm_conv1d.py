#2020-12-04
#embedding(벡터화)
#실습(embedding 빼고 LSTM으로 구성)


from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


#X
docs = ["너무 재밌어요", "참 최고예요", "참 잘 만든 영화예요", "추천하고 싶은 영화입니다",
        "한 번 더 보고 싶네요", "글쎄요", "별로예요", "생각보다 지루해요", "연기가 어색해요",
        "재미없어요", "너무 재미없다", "참 재밌네요"] 

#Y
#1. 긍정 1, 부정 0
labels = np.array([1, 1, 1, 1,1, 0, 0, 0, 0, 0, 0, 1])

#docs를 수치화하면 처리 가능 

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index) #많이 나오는 애가 앞으로 감(빈도수 높은 애): 25개

x = token.texts_to_sequences(docs)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]
# 긴 쪽으로 맞추자 (짧은 쪽에 맞추면 데이터 날아감) 빈 앞을 0으로 채우기(의미 있는 숫자가 뒤로 밀림) 
# 일정하지 않음

from tensorflow.keras.preprocessing.sequence import pad_sequences #0으로 채우는
pad_x = pad_sequences(x, padding='pre')    #뒤로 채우는 건 post
print(pad_x) #(12, 5) docs의 개수, 5(제일 긴 것)
print(pad_x.shape) #25개 vector화


word_size = len(token.word_index) + 1 #padding
# print("전체 토큰 사이즈: ", word_size) 전체 토큰 사이즈:  25
# 12, 5지만 그 안에 들어가는 종류는 25가지



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, MaxPool1D

pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1], 1)

model = Sequential()
model.add(LSTM(128, input_shape=(5, 1), return_sequences=True))
model.add(Conv1D(128, 2))
model.add(Conv1D(64, 2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()



#명시 안 하면 자동으로 먹힌다
                             #26, 10 -> 이래도 에러 남...
                             #input_length만 잘 맞춰 주면... 잘 돌아가는 거 아닌지 
                             #단, 사전의 개수보다 작게 주면 터짐 

# 원핫인코딩이었다면 25, 25 -> 임베딩 레이어로 하면 25, 10으로 벡터화(10은 그냥 임의로 해도 됨)
# 3차원 input -> LSTM 가능
# model.add(LSTM(32))

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['acc'])

model.fit(pad_x, labels, epochs=100)

#data 적으니까 그냥 이렇게 한다 
acc = model.evaluate(pad_x, labels)[1]
print("acc: ", acc)



'''
acc:  1.0
'''