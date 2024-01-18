from flask import Flask, render_template, request
import gensim
from gensim.models import KeyedVectors
import pickle
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

app = Flask(__name__)

# Correct model loading
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def clean_text(text_input):
    text = text_input.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Load GloVe model
glove_model_path = 'word2vec.txt'
glove_model = KeyedVectors.load_word2vec_format(glove_model_path, binary=False)

class_labels = {0: "Figurative",1: "Irony",2: "Regular",3: "Sarcasm"}

# Vectorization using GloVe
def average_word_vectors(tokens, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    nwords = 0
    for i in range(len(tokens)):
        word = tokens[i]
        if word in model:
            nwords += 1
            feature_vector = np.add(feature_vector, model[word][:num_features])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector  # Return the final vector

def preprocess_and_predict(text_input):
    cleaned_text = clean_text(text_input)
    tokens = word_tokenize(cleaned_text)
    vector = average_word_vectors(tokens, glove_model, 100) 
    vector = np.array(vector).reshape(1, -1)
    prediction = model.predict(vector)
    predicted_class = np.argmax(prediction)
    predicted_class = class_labels.get(predicted_class, "Unknown")
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form[ 'text_input' ]
    predicted_class = preprocess_and_predict(text_input)
    return render_template('result.html', text_input=text_input, predicted_class=predicted_class)

#if __name__ == '__main__':
    #app.run()
