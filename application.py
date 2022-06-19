#import libraries
import re 
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from collections import Counter
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import streamlit as st
import emoji 

#Input Data and Data Cleaning
data = pd.read_csv('../emotion-app/Data Mood Analysis.csv')

stop_words = stopwords.words('english')
cleaned_data = []

#clean text: remove punctuatinos, lowercase, split, stemming, and remove stopwords
for i in range(len(data)):
  review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['Text']) 
  review = review.lower().split() #lowercase dan split

  review = [word for word in review if (word not in stop_words)] 
  review = ' '.join(review) 
    
  cleaned_data.append(review)

data2 = data.copy()
data2['Text'] = cleaned_data
data2.head()

## Model Training 
X = data2[['Text']]
y = data2[['Emotion']]
label_encode = LabelEncoder().fit(y)
y = label_encode.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

vocab_size = 5000
embedding_dim = 16
max_length = 128
trunc_type ='pre'
padding_type ='post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok) 
tokenizer.fit_on_texts(X_train['Text']) 
word_index = tokenizer.word_index 
sequences = tokenizer.texts_to_sequences(X_train['Text']) 
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type) 

#for data testing
testing_sequences = tokenizer.texts_to_sequences(X_test['Text']) 
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

model = tf.keras.models.load_model(filepath='./mood-app-python/Mood Analysis Model.h5')

emotions_emoji_dict = {"joy" : emoji.emojize(":joy:"), "sadness" : emoji.emojize(":cry:"), "anger" : emoji.emojize(":angry:"), "fear" : emoji.emojize(":fearful:"), "love" :emoji.emojize(":heart:") , "surprise" : emoji.emojize(":open_mouth:") }

#Functions for application

def predict_emotions(sentence):
    #text cleaning
    sentence = re.sub('[^a-zA-Z]', ' ', sentence) 
    sentence = sentence.lower().split() 

    sentence = [word for word in sentence if (word not in stop_words)] 
    sentence = ' '.join(sentence) 

    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=max_length, truncating=trunc_type)
    result = label_encode.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]

    return result

def get_prediction_proba(sentence):
    #text cleaning
    sentence = re.sub('[^a-zA-Z]', ' ', sentence) 
    sentence = sentence.lower().split() 

    sentence = [word for word in sentence if (word not in stop_words)] 
    sentence = ' '.join(sentence) 

    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=max_length, truncating=trunc_type)
    proba =  np.max(model.predict(sentence))
    return proba * 100



def main():
  menu = ["Home", "About"]
  choice = st.sidebar.selectbox("Menu", menu)

  if choice == "Home":
    st.subheader("Home-Emotion In Text")

    with st.form(key = 'emotion_clf_form'):
      raw_text = st.text_area("Type Here")
      submit_text = st.form_submit_button(label='Submit')

    if submit_text: 
      col1, col2 = st.columns(2)

      prediction = predict_emotions(raw_text)
      probability = get_prediction_proba(raw_text)

      with col1:
        st.success("Original Text")
        st.write(raw_text) 
        
        st.success("Prediction")
        emoji_icon = emotions_emoji_dict[prediction]
        st.write("{}:{}".format(prediction, emoji_icon))
      with col2:
        st.success("Prediction Probability")
        st.write(probability)

  else:
    st.subheader("About")
    st.write("This is a simple web app that applies natural language processing (NLP) tools to detect emotions in sentence and deployed using streamlit. This web app is owned by Valentio Iverson.")

if __name__ == '__main__':
  main()