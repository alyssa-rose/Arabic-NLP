# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:16:30 2019

@author: alyssa rose
"""
import numpy as np
import pandas as pd

df_poem = pd.read_csv("all_poems.csv")
poems = df_poem.iloc[:,3]
poems = poems.astype(str)

  
#import gensim 
#from gensim.models import Word2Vec 

#from bs4 import BeautifulSoup
#import requests
#from csv import writer
#
## web crawling
#response = requests.get("https://www.poetrytranslation.org/poems/in/arabic")
#soup = BeautifulSoup(response.text, 'html.parser')
#
## getting all links on the first page
#links = []
#import re
#for link in soup.find_all('a', attrs={'href': re.compile("^https://")}):
#   links.append(link.get('href'))
#
## crawling through the other pages
#sep_page = []
#link_append = "https://www.poetrytranslation.org/poems/in/arabic"
#increment = ['P12', 'P24', 'P36', 'P48', 'P60', 'P72']
#for i in increment:
#    response = requests.get(link_append+"/"+i)
#    soup = BeautifulSoup(response.text, 'html.parser')
#    for link in soup.find_all('a', attrs={'href': re.compile("^https://")}):
#        sep_page.append(link.get('href'))
#
#poems = []
#for link in sep_page:
#    if(("/poems" in link) and ("/in/" not in link)):
#        if(("/tagged" not in link) and ("/all" not in link) and ("starting-with" not in link)):
#            print(link)
#            link += "/original"
#            poems.append(link)
#content = []
#for poem in poems:
#    response = requests.get(poem)
#    soup = BeautifulSoup(response.text, 'html.parser')
#    body = soup.find(class_ = "poemBody")
#    text = body.find_all('div')
#    encoded = body.find_all('p')
#    for test in text:
#        content.append(test.get_text().replace('\n', ''))
#    for encode in encoded:
#        content.append(encode.get_text().replace('\n', ''))
#
## filtering out unnecessary data
#content[:] = [x for x in content if x!='']
#content[:] = [x for x in content if x!='Share this poemTweet']
#content[:] = [x for x in content if x!='Tweet']
#content[:] = [x for x in content if x!='\xa0']
#

#[x.encode('utf-8') for x in poems]
for i in range(0,len(poems)):
    poems[i] = poems[i].translate({ord(c): None for c in '*-0123456-789]\[,_()\"`\'/><'})
data = '\n'.join(poems[2600:3000])

file = open("arabic_poems.txt", "w", encoding="utf-8")
import re
file.write(data)
file.close()

#fileSeg = open("arabic_segment.txt", "r", encoding="utf-8")
#segLines = fileSeg.readlines()
#fileSeg.close()

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

fileSeg = open("arabic_poems.txt", "r", encoding="utf-8")
segLines = fileSeg.readlines()
fileSeg.close()
from nltk.tokenize import sent_tokenize, word_tokenize
words = word_tokenize(segLines[0])
#new = sent_tokenize(j)
#from gensim.models import Word2Vec
#segModel = Word2Vec([j], min_count=1, size = 50, 
#                        window = 5, sg = 1)
#vec = []
#for i in j:
#    vec.append(segModel[i])
#    
#from collections import Counter
#numWordsUnique = len(Counter(j).keys())
#freqWordsUnique = Counter(j).values()
#
## most similar word
#uniqueWords = Counter(j).keys()
#y = []
#words = []
#
#for i in uniqueWords:
#    words.append(i)
#    temp = segModel.similar_by_word(i)
#    y.append(temp[0][0])
#
#seq_length = 100
#dataX = []
#dataY = []
#for i in range(0, len(vec) - seq_length, 1):
#    seq = vec[i:i + seq_length]
#    seqOut = vec[i + seq_length]
#    dataX.append(seq)
#    dataY.append(seqOut)
#n_patterns = len(dataX)
#
#uniqueVec = []
#for i in words:
#    uniqueVec.append(segModel[i])
#    
#x = np.reshape(dataX, (n_patterns, seq_length, 50))
#y = np_utils.to_categorical(dataY)
#y = np.reshape(dataY, (len(dataY), 50))
#from tensorflow.python.ops import control_flow_ops
#
#orig_while_loop = control_flow_ops.while_loop
#
#def patched_while_loop(*args, **kwargs):
#    kwargs.pop("maximum_iterations", None)  # Ignore.
#    return orig_while_loop(*args, **kwargs)
#
#
#control_flow_ops.while_loop = patched_while_loop


""" REFORMAT """
""" USE OF TOKENIZER, NOT GENSIM WORD2VEC """
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras import utils as np_utils
from numpy import array
from pickle import dump
from pickle import load
from keras.models import load_model
#seq_length = 100
#to_seq = list()
#y = list()


#token.fit_on_texts(j)

"""
test section
"""
token = Tokenizer()
corpus = data.split("\n")
token.fit_on_texts(corpus)
tot_words = len(token.word_index) + 1

token = load(open('token.pkl', 'rb'))
model = load_model('model.h5')
input_seq = []
for line in corpus:
    token_list = token.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_seq.append(n_gram)

max_seq = max([len(x) for x in input_seq])
input_seq = np.array(pad_sequences(input_seq, max_seq, padding = 'pre'))
pred, label = input_seq[:,:-1], input_seq[:,-1]

label = np_utils.to_categorical(label, num_classes = tot_words)

model = Sequential()
model.add(Embedding(tot_words, 50, input_length=max_seq-1))
#model.add(LSTM(256, return_sequences=True))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(tot_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

model.fit(pred, label, batch_size = 256, epochs=50, verbose=1)
model.save('model.h5')

dump(token,open('token.pkl', 'wb'))
words = word_tokenize(data)
length = 100 + 1
sequences = []
for i in range(length, len(words)):
    seq = words[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)
    
for line in range(0, len(sequences)):
    sequences[line] = token.texts_to_sequences([sequences[line]])[0]
#sequences = token.texts_to_sequences(sequences)
max_seq = max([len(x) for x in sequences])
sequences = np.array(pad_sequences(sequences, max_seq, padding = 'pre'))
#sequences=array(sequences)
pred, label = sequences[:,:-1], sequences[:,-1]
label = np_utils.to_categorical(label, num_classes = tot_words)



def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w', encoding = 'utf-8')
	file.write(data)
	file.close()
 
    
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
out_file = "arabic_save.txt"
save_doc(sequences, out_file)

in_file = "arabic_save.txt"
doc = load_doc(in_file)
lines = doc.split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(words)
lines = tokenizer.texts_to_sequences(lines)
#
#min_val = len(lines[0])
#indx = 0
#for i in range(1,len(lines)):
#    if len(lines[i]) < min_val:
#        min_val = len(lines[i])
#        indx = i
#
#new_X = []
#new_Y = []
#for i in range(0, len(sequences)):
#    new_seq = lines[i]
#    new_X.append(new_seq[0:length-1])
#    new_Y.append(new_seq[length-1])
#    
#np.reshape(lines, (len(lines), length))
#np.reshape(new_X, (len(new_X), 50))
#np.reshape(new_Y, (len(new_Y), 1))
#
#new_X = array(new_X)
#vocab = len(tokenizer.word_index) + 1
#new_Y = array(new_Y)
#new_Y = np_utils.to_categorical(new_Y, num_classes=vocab)
#seq_length = new_X.shape[1]

model = Sequential()
model.add(Embedding(vocab, 10, input_length = seq_length))
model.add(LSTM(150))
model.add(Dropout(0.1))
model.add(Dense(vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pred, label, batch_size=500, epochs=50, verbose = 1)

from random import randint
seed_text = array(pred[randint(0, len(pred))])
seed_text = seed_text.reshape(1,100)
#seed_text = word_tokenize(seed_text)
#yhat = model.predict_classes(seed_text)
#out = ''
#for word,index in tokenizer.word_index.items():
#    if index == yhat:
#        out = word
#        break

def get_word(yhat, tokenizer):
    out = ''
    for word,index in tokenizer.word_index.items():
        if index == yhat:
            out = word
            return(out)
        
def print_these_bois(model, tokenizer, seed_text, length_poem, seq_length):
    start_text = ''
    res = list()
    for i in range(0, seed_text.shape[1]):
        start_text += " " + get_word(seed_text[0, i], tokenizer)
    #seed_text = tokenizer.texts_to_sequences([start_text])
    for i in range(0, length_poem):
        encode = tokenizer.texts_to_sequences([start_text])[0]
        print(encode)
        encode = pad_sequences([encode], maxlen = seq_length, truncating='pre')
        prob = model.predict_classes(encode)
        print(get_word(prob, tokenizer))
        start_text += " " + get_word(prob, tokenizer)
        res.append(get_word(prob, tokenizer))
    return ' '.join(res)
        
new_poem = print_these_bois(model, token, seed_text, 75, 100)



"""
"""
rows, cols = (len(sequences), seq_length)
arr = [[0 for j in range(cols)]for i in range(rows)]
np.reshape(arr, (42736, 100))   
        
for i in range(seq_length, len(j)):
    seq = token.texts_to_sequences(j[i - seq_length:i])
    print(seq)
    #seqOut = j[i + seq_length]
    #new_seq = ' '.join(seq)
    #to_seq.append(new_seq)
    #y.append(seqOut)
    for k in range(0, len(seq)):
        arr[i][k] = seq[k]
    
seq_length = 100
dataX = []
dataY = []
for i in range(0, len(vec) - seq_length, 1):
    seq = vec[i:i + seq_length]
    seqOut = vec[i + seq_length]
    dataX.append(seq)
    dataY.append(seqOut)






sequences = token.texts_to_sequences(to_seq)
y = token.texts_to_sequences(y)


sequences = np.asarray(sequences)
sequences.reshape(len(sequences), seq_length-1)
sequences = array(sequences)
new_X = []
new_Y = []
#X, y = sequences[:,:1], sequences[:,1]
for i in range(0, len(sequences)-1):
    k = sequences[i]
    print(len(k))
    new_X.append(k[0:seq_length-1])
    new_Y.append(k[seq_length-1])
    #sequences[i] = array(sequences[i])
sequences = sequences.reshape(sequences.shape[0], 100)
vocab = len(token.word_index) + 1


#sequences = np.asarray(sequences)
x, y = sequences[:,:-1], sequences[:-1]
y = array(y)

new_y = []
for i in range(0, len(y)-1):
    new_y.append(np_utils.to_categorical(y[i], num_classes = vocab))
#y = np_utils.to_categorical(y, num_classes = vocab)
sequences = array(sequences)
seq_length = len(sequences[0])

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(Embedding(vocab, 50, input_length = 1))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(sequences, y, batch_size=128, epochs=25)

"""""""
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x, y, epochs=20, batch_size = 128, verbose=5) 
uniqueVec = list(uniqueVec)
words = list(words)

def vec_to_word(vector):
    for i in range(0,len(uniqueVec)):
        if np.linalg.norm(uniqueVec[i]-vector) < .09:
            return words[i]
#        else :
#            print(np.linalg.norm(uniqueVec[i]-vector) )
#        
    

start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
results = []

for i in range(20):
    X = np.reshape(pattern, (1, len(pattern), 50))
    pred = model.predict(X, verbose = 0)
    print(pred)
    result = vec_to_word(pred)
    results.append(result)
    seq_in.append([vec_to_word(val) for val in pattern])
    #sys.stdout.write(result)
    pattern.append(np.reshape(pred, (50,)))
    pattern = pattern[1:len(pattern)]
    print(i)
    
if(np.argmax(predict) in uniqueVec):
    print("okie")
#import fastText
#model = Sequential()
#model.add(LSTM(75, input_shape = (X.shape[1], X.shape[2])))
#model.add(Dense(vocab_size, activation = 'softmax'))
#
#model.compile(loss = 'categorical_crossentropy'
#              optimizer = 'adam',
#              metrics = ['accuracy'])
#
#model.fit(X, y, epochs = 100, verbose = 2)