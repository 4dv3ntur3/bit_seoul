#2020-12-04
#embedding
#자연어 처리 + tokenizing

from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 맛있는 밥을 진짜 먹었다.'
token = Tokenizer()

#token에 text 적용
token.fit_on_texts([text])

#어절별로 수치가 먹혔다
#많이 나오는 단어가 앞 index 가져감 

print(token.word_index)
print(len(token.word_index))
# {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}

x = token.texts_to_sequences([text])
print(x) #[[2, 1, 3, 4, 1, 5]]

#'나는'은 '진짜'의 두 배? x
# ONE HOT ENCODING
from tensorflow.keras.utils import to_categorical #0부터 one hot encoding 해 줘서
word_size = len(token.word_index) 
x = to_categorical(x, num_classes=word_size+1) #지금이야 적으니까 괜찮지만 10000개면 10000*10000...
print(x)


