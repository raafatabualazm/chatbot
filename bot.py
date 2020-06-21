import nltk
import tensorflow as tf
import numpy as np
import json
import random
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import pickle

vectorizer = CountVectorizer()
stemmer = SnowballStemmer('english')
enc = OneHotEncoder()
# nltk.download("punkt")
conv_file = open("conv.json")
conv_data = json.load(conv_file)


def get_bow(s, word_list):
    bow = [0 for _ in range(len(word_list))]
    s_words = nltk.word_tokenize(s)
    for word in s_words:
        word_stem = stemmer.stem(word.lower())
        for i, w in enumerate(word_list):
            if w == word_stem:
                bow[i] += 1
    return np.array(bow).reshape(1, 42)
    

def chat(word_list, tags_list):
    print("Welcome! Type quit or exit to end the program.")
    while True:
        inp = input("You: ")
        if inp.lower == 'quit' or inp.lower() == 'exit':
            break
        else:
            bow = get_bow(inp, word_list)
            result = model.predict(bow)
            print(result.max())
            if result.max() >= 0.35:
                tag = tags_list[np.argmax(result)]
                for intent in conv_data['intents']:
                    if intent['tag'] == tag[3:]:
                        print(random.choice(intent['responses']))
                        break
            else:
                print("I did not understand what you were saying. Please try again.")


try:
    model = tf.keras.models.load_model('bot.hdf5')
    infile = open('word_list', 'rb')
    word_list = pickle.load(infile)
    infile2 = open('tags_list', 'rb')
    tags_list = pickle.load(infile2)
    infile.close()
    infile2.close()
    chat(word_list, tags_list)
except:

    docs_stem = []
    docs_tags = []
    for intent in conv_data["intents"]:
        for pattern in intent["patterns"]:
            pattern_stem = ''
            for word in nltk.word_tokenize(pattern):
                stemw = stemmer.stem(word.lower())
                pattern_stem = pattern_stem + " " + stemw
            pattern_stem.strip()
            docs_stem.append(pattern_stem)
            docs_tags.append(intent["tag"])
        

    
    X = vectorizer.fit_transform(docs_stem).toarray()
    y = enc.fit_transform(np.array(docs_tags).reshape(-1, 1)).toarray()
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1])))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(y.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    model.fit(X, y, batch_size=64, epochs=400)

    tags_list = enc.get_feature_names()
    model.save('bot.hdf5')
    word_list = vectorizer.get_feature_names()
    outfile = open('word_list', 'wb')
    pickle.dump(word_list, outfile)
    outfile2 = open('tags_list', 'wb')
    pickle.dump(tags_list, outfile2)
    outfile.close()
    outfile2.close()
    chat(word_list, tags_list)

