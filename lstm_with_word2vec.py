import os
import re
import csv
import sys
import codecs
import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import time
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Global parameters
EMBEDDING_FILE = 'pretrained_model/GoogleNews-vectors-negative300.bin' 
TRAIN_DATA_FILE = 'data/dataset.csv'
MAX_SEQUENCE_LENGTH = 30 # max of words in asentence to make the vectors equal in shape
MAX_NB_WORDS = 200000 # max number of unique words in corpus
EMBEDDING_DIM = 300 # shape of embedding
VALIDATION_SPLIT = 0.1
# Adding some dropout regulsrixzation and units in LSTM and dense layer
num_lstm = 200
num_dense = 125
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

#loading the pretrained word2vec embeddings 
word2vec = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
print("---------------------------------------------------------------------------------------------------------------")
print(" PRETRAINED MODEL LOADED SUCESSFULLY : WORD2VEC")


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    '''
    Input: Text
    Function: Clean the text with the option to remove stopwords and to stem words.
    Returns: A list of words'''
    text = text.lower().split()# Convert words to lower case and split them
    if remove_stopwords:# Remove stop words
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    print("Text to wordlist received: ",text)
    return(text)

# reading the Dataset and splitting the sentences and outputs
texts_1 = [] 
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[1]))
        texts_2.append(text_to_wordlist(values[2]))
        labels.append(int(values[3]))
print('Found %s texts in train.csv' % len(texts_1))

test_senten_1 = ["I lost my credit card at a restaurant"]
test_senten_2 = ["I forgot my credit card at a place"]
test_labels = []

# tokenizing the words in the corpus to pass it in to embedding layer
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_senten_1 + test_senten_2)
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_senten_2)
test_sequences_2 = tokenizer.texts_to_sequences(test_senten_1)


word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
# padding the sequences to make it equal length
data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

# Preparing embedding matrix 
# embedding_matrix contains the embeddings of the words in the corpus
nb_words = min(MAX_NB_WORDS, len(word_index))+1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

# Ndividing train and validation sets
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]
data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))
print("Length of data_train and labels_train")
print(len(data_1_train), len(labels_train))
data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

# model and training
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)
sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation='relu')(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
preds = Dense(1, activation='sigmoid')(merged)
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val), \
        epochs=10, batch_size=2048, shuffle=True)

#predicting and inferencing
preds = model.predict([test_data_1, test_data_2])
preds += model.predict([test_data_2, test_data_1])
preds /= 2
print("Predict:",preds)

test1 = ["Many black dogs run in a grassy area."]
test2 = ["A group of dogs are playing."]
start = time.time()

test_1 = tokenizer.texts_to_sequences(test1)
test_2 = tokenizer.texts_to_sequences(test2)

data_test_1 = pad_sequences(test_1, maxlen=MAX_SEQUENCE_LENGTH)
data_test_2 = pad_sequences(test_2, maxlen=MAX_SEQUENCE_LENGTH)
preds = model.predict([data_test_1, data_test_2])
end = time.time()
#time taken to execute and prediction
print(end-start, preds)

test_1 = input("Enter the text1")
test_2 = input("Enter the text2")
start = time.time()

test_1 = tokenizer.texts_to_sequences(test1)
test_2 = tokenizer.texts_to_sequences(test2)

data_test_1 = pad_sequences(test_1, maxlen=MAX_SEQUENCE_LENGTH)
data_test_2 = pad_sequences(test_2, maxlen=MAX_SEQUENCE_LENGTH)
preds = model.predict([data_test_1, data_test_2])
end = time.time()
print(end-start, preds)
