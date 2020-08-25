# Arabic_NLP
Personal project to use an LSTM to generate novel Arabic poems.

## Setup
  1. Download the repository
  2. In the command window, navigate to the repository such that it's the current working directory
  3. (optional) create a virtual environment to download packages to
  4. run `pip install -r requirements.txt` to download the required packages for the program (Tensorflow issues may arise from an outdated Python version)
  5. run `python arabic.py` after packages finishing dowloading
You will then be prompted for the desired number of poems to generate, and whether or not to use the pre-trained model file or not. Enter 'y' to use the model, or 'n' if you
don't want to. The pre-trained model is stored in the file `model.h5`. If the file doesn't exist, the model will be trained normally. If `arabic_poems.txt` is also in cwd, it
will be used instead of rescraping the website. After running, generated poems will be found in the cwd as `gen_poems.txt`

## Methodolgy
The program is rendered functional via use of the keras & Tensorflow libraries to create an LSTM (long short term memory) model. The model is trained on sequences of lengths 
approximately 50 - 60 words long. The model is fit on the training data with batch_size of 128 and 100 epochs. The model contains an embedding layer, two LSTM layers with 150
nodes each, and two dense layers.

Text is generated using seed text that was already seen by the model. On every iteration of generating a new word, the previous 50 to 60 characters (depending on sequence size)
is used to create a probability distribution across all words in the tokenizer's vocabulary. The highest probability is taken as the most likely next word in the sequence, and 
this process contains length_poems iterations. On every iteration, the sequence is truncated such that previously generated words are used to predict the next word.

## Data
Data was scraped from poetrytranslations.org using the beautifulsoup4 library. Diacritics were cleaned from the text using the PyArabic library.
  
