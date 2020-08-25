# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 03:00:27 2020

@author: alyssa

Script to generate Arabic poems using an LSTM (long short-term memory) model
"""


import os
import re
import nltk
import requests
import numpy as np
nltk.download('punkt')
from bs4 import BeautifulSoup
import pyarabic.araby as araby
from keras.layers import Embedding
from keras import utils as np_utils
from nltk.tokenize import word_tokenize
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences







def scrape_poetry_trans()->list:
    '''
    Returns
    -------
    list
        List of links to all poems contained in the Poetry Translation 
        database for original Arabic poems.
    '''
    
    sep_page = list()
    
    # Original links to main site
    link_append = "https://www.poetrytranslation.org/poems/in/arabic"
    # page numbers to iterate through
    increment = ['','P12', 'P24', 'P36', 'P48', 'P60', 'P72']
    for i in increment:
        # building desired url to specific page number
        response = requests.get(link_append+"/"+i)
        soup = BeautifulSoup(response.text, 'html.parser')
        # finding all links to other sites contained within main page
        for link in soup.find_all('a', attrs={'href': re.compile("^https://")}):
            sep_page.append(link.get('href'))
            
    # keeping only the links to the pages that have the original Arabic poems
    poems = [k+"/original" for k in sep_page if is_poem(k) == True]
    return poems





def is_poem(link: str)-> bool:
    '''
    Parameters
    ----------
    link : str
        Link to a subsite as branched off from pages contained in
        https://www.poetrytranslation.org/poems/in/arabic

    Returns
    -------
    bool
        True if link is to a poem page. False otherwise.

    '''
    if(("/tagged" not in link) and ("/all" not in link) and ("starting-with" not in link)):
        if(("/poems" in link) and ("/in/" not in link)):
                return True
    return False
        
    



def clean_poems(poems: list)-> list:
    '''
    Parameters
    ----------
    poems : list
        List containing all links to the original poems

    Returns
    -------
    list
        Contains filtered text that was gathered from scraping all links
        in the input poems list. Non-essential characters and blank lines
        are removed. Individual poems split into multiple lines.

    '''
    content = list()
    for poem in poems:
        response = requests.get(poem)
        soup = BeautifulSoup(response.text, 'html.parser')
        # html div that contains the actual poem text
        body = soup.find(class_ = "poemBody")
        # text found within <p></p> div
        encoded = body.find_all('p')
        for encode in encoded:
            # changing line breaks to spaces
            content.append(encode.get_text().replace('\n', ' '))
    
    # filtering out additional spaces
    content[:] = [x for x in content if x.strip() !='']
    # this unicode character kept appearing (non break space)
    content[:] = [x for x in content if x!='\xa0']
    
    # removing non-essential characters
    content = [j.translate({ord(c): None for c in '*-!?0123456-789]\[,_()\"`\'/><...'}) for j in content]
    '''
    for i in range(0, len(content)):
       content[i] = content[i].translate({ord(c): None for c in '*-!?0123456-789]\[,_()\"`\'/><...'})
   '''
    return content
 
    



def outfile()-> str:
    '''
    Returns
    -------
    str
        Cleaned corpus of all poems. Poems joined as one text, and diacritics
        removed. Additionally, the clean text is written to a file to prevent 
        having to scrape data over and over.

    '''
    
    # if text file already exists
    try:
        f = open('arabic_poems.txt', encoding="utf-8")
        data = f.read()
        f.close()
    except:
        content = clean_poems(scrape_poetry_trans())
        
        data = ' '.join(content)
        data = araby.strip_diacritics(data)
        file = open("arabic_poems.txt", "w+", encoding="utf-8")
        file.write(data)
        file.close()
   
    
    return data





def create_sequences(data: str, seq_len: int)-> tuple:
    '''
    Parameters
    ----------
    data : str
        Cleaned corpus of all poems.
    seq_len : int
        Desired sequence length to train on. Actual sequence length will be smaller
        due to the texts_to_sequences function

    Returns
    -------
    tuple
        training data (x & y values), the number of unique words in the vocabulary.
        the input size for the model (x.shape[1]), and the tokenizer that trained
        on the sequences.

    '''
    # splitting corpus into individual words
    tokens = word_tokenize(data)
    
    # creating individual lines that are of length seq_len
    sequences = list()
    for i in range(seq_len, len(tokens)):
        seq = tokens[i-seq_len:i]
        line = ' '.join(seq)
        sequences.append(line)
    
    # converting the sequences (str) to integer representations
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    
    # determing the shortest sequence length so that the other sequences
    # will be truncated to that length (so that the model can be trained as
    # it needs a consistent input length)
    min_val = len(sequences[0])
    for k in range(0, len(sequences)):
        if len(sequences[k]) < min_val:
            min_val = len(sequences[k])
    
    
    # truncating the longer sequences to be of length min_val and casting to array
    sequences = np.array(pad_sequences(sequences, maxlen=min_val, truncating='pre'))
    
    # number of unique words contained in the corpus; vocabulary size
    num_vocab = len(set(tokens))
    
    # x has sequence[0:min_val -1] and y has the last value for each given sequence
    x, y = sequences[:,:-1], sequences[:,-1]
    
    # encoding y to be a categorical value based on the options of what word
    # it is from the vocabulary. If it is word 'a', then y[i,a] == 1 and all other
    # values y[i, k!=a] == 0 
    y = np_utils.to_categorical(y, num_classes=num_vocab)
    
    # input size for the model is actually the size of the second shape value
    # of x. Since contains the sequence value except for the last one per 
    # sequence, it's second shape value is actually min_val - 1
    min_val -= 1
    
    return(x, y, num_vocab, min_val, tokenizer)





def run_model(x, y, num_vocab: int, seq_length: int):
    '''
    Parameters
    ----------
    x : numpy.ndarray
        training values
    y : numpy.ndarray
        prediction values
    num_vocab : int
        number of unique words contained in the corpus
    seq_length : int
        Size of the input, == x.shape[1]

    Returns
    -------
    model : Keras model
        Returns the keras sequential LSTM model that was trained on the sequences,
         represented by the x & y input values (x is train, y is the pred value)

    '''
    model = Sequential()
    model.add(Embedding(num_vocab, 50, input_length=seq_length))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(num_vocab, activation='softmax'))
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(x, y, batch_size=128, epochs=100)
    
    # saving to be used later if one doesn't wish to wait for the model to train
    model.save('model.h5')
        
    return model





def get_word(yhat: float, tokenizer)-> str:
    '''
    Parameters
    ----------
    yhat : float
        argmax value of the prediction array (the highest probability)
    tokenizer : keras_preprocessing.text.Tokenizer
        Tokenizer that was trained on the sequence data

    Returns
    -------
    str
        The original unencoded word in Arabic.

    '''
    out = ''
    try:
        for word,index in tokenizer.word_index.items():
            if index == yhat:
                out = word
                return(out)
    # highly unlikely case that the word isn't in the tokenizer dictionary.
    # occured when pad_sequences() used pre-padding instead of truncating
    # which led to '0' being in the sequence, which is not an index in the 
    # tokenizer's word_index
    except:
        print(yhat)
        
        
        
        
        
def gen_poem(model, tokenizer, seed_text, length_poem: int, seq_length: int)-> tuple:
    '''
    Parameters
    ----------
    model : Keras model
        the trained Keras sequential LSTM model
    tokenizer : keras_preprocessing.text.Tokenizer
        Tokenizer that was trained on the sequence data
    seed_text : numpy.ndarray
        Tokenizer encoded text of length seq_length that is used to start
        generating words off of. Text that was already seen by the model.
    length_poem : int
        Desired length of the generated poem.
    seq_length : int
        input length that was used to train the model, == min_val

    Returns
    -------
    tuple
        seed + generated text and just the generated text

    '''
    start_text = ''
    total_gen = list()
    for i in range(0, len(seed_text)):
        # convert the encoded seed text to words
        start_text += " " + get_word(seed_text[i], tokenizer)
        
    for i in range(length_poem):
        # encode the entire text, which will contain the seed and the generated
        encode = tokenizer.texts_to_sequences([start_text])[0]
        # truncate to seq_length, removing from the front
        encode = pad_sequences([encode], maxlen = seq_length, truncating='pre')
        
        # REPLACEMENT FOR OUTDATED PREDICT_CLASSES
        # probability distributiont that each word in vocabulary is the next
        # word that should come in the encode sequence. Take the highest probability.
        predicting = model.predict(encode)
        prob = np.argmax(predicting)
        
        # convert the encoded word to Arabic text
        out = get_word(prob, tokenizer)
        start_text += " " + out
        total_gen.append(out)
        
    return start_text, ' '.join(total_gen)







def main(num_poems: int, model_bool: str, sequence_len=60)->list:
    '''
    Parameters
    ----------
    num_poems : int
        Number of poems to generate, inputted by the user
    model_bool : str
        'y' to use pre-trained model if available, 'n' otherwise
    Returns
    -------
    list
        list of generated poems.

    '''
    data = outfile()
    x, y, num_vocab, min_val, tokenizer = create_sequences(data, sequence_len)
    
    if model_bool == 'y':
        try:
            model = load_model('model.h5')
        except:
            print('No available model file')
            model = run_model(x, y, num_vocab, min_val)
    else:
        model = run_model(x, y, num_vocab, min_val)

    poems = list()
    for i in range(num_poems):
        # select random poem from poem list to start generating a new
        # poem off of
        start = np.random.randint(0, x.shape[1]-1)
        seed_text = x[start]
        whole, gen = gen_poem(model, tokenizer, seed_text, 65, min_val)
        # keeping only the generated portion and not the seed text
        poems.append(gen)
        
    return poems





if __name__ == "__main__":
    print("Hit ENTER after typing in values\n")
    print("Number of poems to generate: ")
    num_poems = int(input())
    print("Use pre-existing model? y/n: ")
    model_bool = (input()).lower()
    if model_bool != 'y' and model_bool != 'n':
        print("Enter y or n: ")
        model_bool = (input()).lower()
    poems = main(num_poems, model_bool)
    f = open('gen_poems.txt', 'w+', encoding='utf-8')
    poems = '\n\n'.join(poems)
    f.write(poems)
    f.close()
    print("Poems written to 'gen_poems.txt'")
