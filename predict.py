# predict.py

import tensorflow as tf
import numpy as np
from nltk.tokenize import sent_tokenize

# Example function for sentence tokenization
def get_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Function to predict the abstract's headings and organize the sentences
def predict_and_organize(text: str, model: tf.keras.Model):
    sentences = get_sentences(text)
    predictions = []
    
    for sentence in sentences:
        input_data = np.array([sentence])
        prediction = model.predict(input_data)
        predictions.append(prediction)
    
    # Mapping predictions to the corresponding class names
    classes = {
        0: 'BACKGROUND',
        1: 'CONCLUSIONS',
        2: 'METHODS',
        3: 'OBJECTIVE',
        4: 'RESULTS'
    }
    
    def get_class(prediction: list):
        return classes[np.argmax(prediction)]
    
    # Organize sentences by class
    organized = {key: [] for key in classes.values()}
    for sentence, prediction in zip(sentences, predictions):
        heading = get_class(prediction)
        organized[heading].append(sentence)
    
    return organized
