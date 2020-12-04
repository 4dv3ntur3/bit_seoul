#2020-12-04
#embedding(벡터화)

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
print(token.word_index) #많이 나오는 애가 앞으로 감(빈도수 높은 애)

x = token.texts_to_sequences(docs)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]
# 긴 쪽으로 맞추자 (짧은 쪽에 맞추면 데이터 날아감) 빈 앞을 0으로 채우기(의미 있는 숫자가 뒤로 밀림) 
# 일정하지 않음

from tensorflow.keras.preprocessing.sequence import pad_sequences #0으로 채우는
pad_x = pad_sequences(x, padding='pre')    #뒤로 채우는 건 post
print(pad_x)
