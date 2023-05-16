import csv
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split


def truncate(arr, size):
    if len(arr) > size:
        return arr[:size + 1]
    else:
        copy = arr.copy()
        copy += (size + 1 - len(arr)) * ['<pad>']
        return copy


def preprocess_sentence_list(sentence_list, window_size):
    for i, sentence in enumerate(sentence_list):
        sentence_new = ['<start>'] + sentence[:window_size - 1] + ['<end>']
        sentence_list[i] = truncate(sentence_new, window_size)



def preprocess_data_to_X_Y_for_tokenization(data):
    X = []
    Y = []
    for dish, ingredients in data.items():
        X.append(dish)
        Y.append(ingredients)
    return X, Y


def regex_tokenization_for_X_Y(X, Y, save_to_file=False):
    non_alpha_num = re.compile('[^a-zA-Z0-9 ]')

    for i, (x, y) in enumerate(zip(X, Y)):
        x = x.lower()
        x = re.sub(non_alpha_num, '', x)
        x_tokens = re.split('\s+', x)
        X[i] = x_tokens

        for j, ingredient in enumerate(y):
            ingredient = ingredient.lower()
            ingredient = re.sub(non_alpha_num, '', ingredient)
            ingredient_tokens = re.split('\s+', ingredient)
            ingredient = ' '.join(ingredient_tokens)
            y[j] = ingredient

    data_to_dump = {'X': X, 'Y': Y}

    if save_to_file:
        with open('data_lemmatized.p', 'wb') as f:
            pickle.dump(data_to_dump, f)
    return X, Y


def spacy_tokenization_for_X_Y(X, Y, save_to_file=False):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    total = len(X)
    for i, (x, y) in enumerate(zip(X, Y)):
        print(f'\r {i + 1}/{total} spacy lemmatization', end='')
        x_tokens = nlp(x)
        x_lemmatized = []
        for token in x_tokens:
            if not token.is_punct:
                x_lemmatized.append(token.lemma_.lower())
        X[i] = x_lemmatized

        for j, ingredient in enumerate(y):
            ingredient_tokens = nlp(ingredient)
            ingredient_lemmatized = []
            for token in ingredient_tokens:
                if not token.is_punct:
                    ingredient_lemmatized.append(token.lemma_.lower())
            ingredient = ' '.join(ingredient_lemmatized)
            y[j] = ingredient
    print()
    data_to_dump = {'X': X, 'Y': Y}

    if save_to_file:
        with open('data_lemmatized.p', 'wb') as f:
            pickle.dump(data_to_dump, f)
    return X, Y


def preprocess_paired_data(data=None, window_size=20, save_to_file=False, file_name='prep_data.p', data_size=None, min_frequency=50):
    if data is None:
        X = [['Glazed', 'Finger', 'Wings'],
                   ['Country', 'Scalloped', 'Potatoes', '&amp;', 'Ham', '(Crock', 'Pot)'],
                   ['Fruit', 'Dream', 'Cookies'], ['Tropical', 'Breakfast', 'Risotti'],
                   ['Linguine', 'W/', 'Olive,', 'Anchovy', 'and', 'Tuna', 'Sauce']]
        Y = [['chicken-wings', 'sugar,', 'cornstarch', 'salt', 'ground ginger', 'pepper', 'water', 'lemon juice',
                    'soy sauce'],
                   ['potatoes', 'onion', 'cooked ham', 'country gravy mix', 'cream of mushroom soup', 'water',
                    'cheddar cheese'],
                   ['butter', 'shortening', 'granulated sugar', 'eggs', 'baking soda', 'baking powder', 'vanilla',
                    'all-purpose flour', 'white chocolate chips', 'orange drink mix', 'colored crystal sugar'],
                   ['water', 'instant brown rice', 'pineapple tidbits', 'skim evaporated milk', 'raisins',
                    'sweetened flaked coconut', 'toasted sliced almonds', 'banana'],
                   ['anchovy fillets', 'tuna packed in oil', 'kalamata olive', 'garlic cloves', 'fresh parsley',
                    'fresh lemon juice', 'salt %26 pepper', 'olive oil', 'linguine']]
    else:
        X = data[0]
        Y = data[1]

    if data_size is not None:
        X = X[:data_size]
        Y = Y[:data_size]

    ingredient_word2idx = {}
    dish_word2idx = {}
    ingredient_vocab_size = 0
    dish_vocab_size = 0

    preprocess_sentence_list(X, window_size)
    preprocess_sentence_list(Y, window_size)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = 0.99,random_state=42)
    src_word_count = Counter()
    tgt_word_count = Counter()

    for sentence in X_train:
        src_word_count.update(sentence)
    for sentence in Y_train:
        tgt_word_count.update(sentence)


    def unk_process(sentence_list, counter, min_frequency):
        unk = 0
        for i, sentence in enumerate(sentence_list):
            for j, word in enumerate(sentence):
                if counter[word] <= min_frequency:
                    sentence[j] = '<unk>'
                    unk += counter[word]
        return unk

    unk_src = unk_process(X_train, src_word_count, min_frequency)
    unk_tgt = unk_process(Y_train, tgt_word_count, min_frequency)

    print(f'unk_src: {unk_src}, unk_tgt: {unk_tgt}')
    print(f'total src words: {sum(src_word_count.values())}, total tgt words: {sum(tgt_word_count.values())}')

    for x, y in zip(X_train, Y_train):
        for word in x:
            if word not in dish_word2idx:
                dish_word2idx[word] = dish_vocab_size
                dish_vocab_size += 1
        for word in y:
            if word not in ingredient_word2idx:
                ingredient_word2idx[word] = ingredient_vocab_size
                ingredient_vocab_size += 1

    dish_idx2word = {i: w for w, i in dish_word2idx.items()}
    ingredient_idx2word = {i: w for w, i in ingredient_word2idx.items()}

    for i, (x, y) in enumerate(zip(X_train, Y_train)):
        X_train[i] = [dish_word2idx[word] for word in x]
        Y_train[i] = [ingredient_word2idx[word] for word in y]

    data_to_dump = {'X': X_train,
                    'Y': Y_train,
                    'X_test': X_test,
                    'Y_test': Y_test,
                    'dish_word2idx': dish_word2idx,
                    'dish_idx2word': dish_idx2word,
                    'dish_vocab_size': dish_vocab_size,
                    'ingredient_word2idx': ingredient_word2idx,
                    'ingredient_idx2word': ingredient_idx2word,
                    'ingredient_vocab_size': ingredient_vocab_size,
                    'window_size': window_size
                    }


    if save_to_file:
        with open(file_name, 'wb') as processed_file:
            pickle.dump(data_to_dump, processed_file)

    return data_to_dump


def pipeline_for_tokenization(data, use_spacy=False, save_to_file=False):

    X, Y = preprocess_data_to_X_Y_for_tokenization(data)
    
    lemmatizer = spacy_tokenization_for_X_Y if use_spacy else regex_tokenization_for_X_Y
    X, Y = lemmatizer(X, Y, save_to_file=save_to_file)
    
    prep_data = preprocess_paired_data(data=(X, Y), save_to_file=save_to_file)

    return prep_data



def get_data_p():
    with open('data.p', 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    pipeline_for_tokenization(data, use_spacy=use_spacy, save_to_file=save_to_file)
