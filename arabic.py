# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 03:00:27 2020

@author: alyssa

Script to generate Arabic poems using an LSTM (long short-term memory) model
"""

import numpy as np
import os
from bs4 import BeautifulSoup
import requests
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras import utils as np_utils
from pickle import dump

import pyarabic.araby as araby


def scrape_poetry_trans()->list:
    sep_page = []
    link_append = "https://www.poetrytranslation.org/poems/in/arabic"
    increment = ['P12', 'P24', 'P36', 'P48', 'P60', 'P72']
    for i in increment:
        response = requests.get(link_append+"/"+i)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', attrs={'href': re.compile("^https://")}):
            sep_page.append(link.get('href'))

    poems = []
    for link in sep_page:
        if(("/poems" in link) and ("/in/" not in link)):
            if(("/tagged" not in link) and ("/all" not in link) and ("starting-with" not in link)):
                link += "/original"
                poems.append(link)
                
    return poems
            




def clean_poems(poems:list)->list:
    content = []
    for poem in poems:
        response = requests.get(poem)
        soup = BeautifulSoup(response.text, 'html.parser')
        body = soup.find(class_ = "poemBody")
        encoded = body.find_all('p')
        for encode in encoded:
            content.append(encode.get_text().replace('\n', ''))
    
    # filtering out unnecessary data
    content[:] = [x for x in content if x!='']
    content[:] = [x for x in content if x!='\xa0']
    
    for i in range(0,len(content)):
       content[i] = content[i].translate({ord(c): None for c in '*-!?0123456-789]\[,_()\"`\'/><...'})
       
    return content
 
    



def outfile():
    content = clean_poems(scrape_poetry_trans())
    os.chdir(r'C:\Users\alyss\Documents\Projects\Arabic_NLP')
    
    data = ' '.join(content)
    data = araby.strip_diacritics(data)
    file = open("arabic_poems.txt", "w+", encoding="utf-8")
    file.write(data)
    file.close()
   
    
    return data





def create_sequences(data, seq_len):
    tokens = word_tokenize(data)

    sequences = []
    for i in range(seq_len, len(tokens)):
        seq = tokens[i-seq_len:i]
        line = ' '.join(seq)
        sequences.append(line)
    
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    
    min_val = len(sequences[0])
    for k in range(1, len(sequences)):
        if len(sequences[k]) < min_val:
            min_val = len(sequences[k])
        
    sequences = pad_sequences(sequences, maxlen=min_val, truncating='pre')
    
    sequences = np.array(sequences)
    num_vocab = len(set(tokens))
    
    x, y = sequences[:,:-1], sequences[:,-1]
    y = np_utils.to_categorical(y, num_classes=num_vocab)
    
    return(x, y, num_vocab, min_val, tokenizer)





def run_model(x, y, num_vocab, seq_length):
    model = Sequential()
    model.add(Embedding(num_vocab, 50, input_length=seq_length))
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(num_vocab, activation='softmax'))
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(x, y, batch_size=128, epochs=100)
    
    
    model.save('model_150.h5')
        
    return model





def get_word(yhat, token):
    out = ''
    try:
        for word,index in token.word_index.items():
            if index == yhat:
                out = word
                return(out)
    except:
        print(yhat)
        
        
        
        
        
def gen_poem(model, tokenizer, seed_text, length_poem, seq_length):
    start_text = ''
    total_gen = []
    for i in range(0, len(seed_text)):
        start_text += " " + get_word(seed_text[i], tokenizer)
    #seed_text = tokenizer.texts_to_sequences([start_text])
    for i in range(length_poem):
        encode = tokenizer.texts_to_sequences([start_text])[0]
        # print(encode)
        encode = pad_sequences([encode], maxlen = seq_length, truncating='pre')
        
        # REPLACEMENT FOR OUTDATED PREDICT_CLASSES
        predicting = model.predict(encode)
        prob = np.argmax(predicting)
        
        out = get_word(prob, tokenizer)
        start_text += " " + out
        total_gen.append(out)
    return start_text, ' '.join(total_gen)







def main(num_poems):
    data = outfile()
    x, y, num_vocab, min_val, tokenizer = create_sequences(data, 60)
    model = run_model(x, y, num_vocab, min_val)

    poems = []
    for i in range(num_poems):
        start = np.random.randint(0, x.shape[1]-1)
        seed_text = x[start]
        whole, gen = gen_poem(model, tokenizer, seed_text, 50, min_val)
        poems.append(whole)
        
    return poems





if __name__ == "__main__":
    print("Number of poems to generate: ")
    num_poems = int(input())
    poems = main(num_poems)
    f = open('gem_poems.txt', 'w+', encoding='utf-8')
    poems = '\n\n'.join(poems)
    f.write(poems)
    f.close()
    print("Poems written to 'gen_poems.txt'")
