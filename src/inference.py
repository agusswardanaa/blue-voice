import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
from tensorflow.keras.models import load_model

# Fungsi untuk text preprocessing
def cleaning_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # menghapus mention
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # menghapus hashtag
    text = re.sub(r'RT[\s]', '', text) # menghapus RT
    text = re.sub(r"http\S+", '', text) # menghapus link
    text = re.sub(r'[0-9]+', '', text) # menghapus angka
    text = re.sub(r'[^\w\s]', '', text) # menghapus karakter selain huruf dan angka
 
    text = text.replace('\n', ' ') # mengganti baris baru dengan spasi
    text = text.translate(str.maketrans('', '', string.punctuation)) # menghapus semua tanda baca
    text = text.strip(' ') # menghapus karakter spasi dari kiri dan kanan teks
    return text
 
def casefolding_text(text): # Mengubah semua karakter dalam teks menjadi huruf kecil
    text = text.lower()
    return text
 
def tokenizing_text(text): # Memecah atau membagi string, teks menjadi daftar token
    text = word_tokenize(text)
    return text
 
def filtering_text(text): # Menghapus stopwords dalam teks
    listStopwords = set(stopwords.words('english'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text
 
def to_sentence(list_words): # Mengubah daftar kata menjadi kalimat
    sentence = ' '.join(word for word in list_words)
    return sentence

# Load model
tfidf_attractions = joblib.load("models/tfidf_attractions.pkl")
tfidf_amenities = joblib.load("models/tfidf_amenities.pkl")
tfidf_access = joblib.load("models/tfidf_access.pkl")
tfidf_price = joblib.load("models/tfidf_price.pkl")
tfidf_no_aspect = joblib.load("models/tfidf_no_aspect.pkl")

model_attractions = load_model("models/nn_attractions.h5")
model_amenities = load_model("models/nn_amenities.h5")
model_access = load_model("models/nn_access.h5")
model_price = load_model("models/nn_price.h5")
model_no_aspect = load_model("models/nn_no_aspect.h5")

# Fungsi untuk inferensi
def predict_sentiment(text):
    # Preprocessing
    text = cleaning_text(text)
    text = casefolding_text(text)
    text = tokenizing_text(text)
    text = filtering_text(text)
    text = to_sentence(text)

    # Transformasi teks ke dalam bentuk TF-IDF
    tfidf_attractions_text = tfidf_attractions.transform([text])
    tfidf_amenities_text = tfidf_amenities.transform([text])
    tfidf_access_text = tfidf_access.transform([text])
    tfidf_price_text = tfidf_price.transform([text])
    tfidf_no_aspect_text = tfidf_no_aspect.transform([text])

    # Prediksi
    prediction_attractions = model_attractions.predict(tfidf_attractions_text, verbose=0)
    prediction_amenities = model_amenities.predict(tfidf_amenities_text, verbose=0)
    prediction_access = model_access.predict(tfidf_access_text, verbose=0)
    prediction_price = model_price.predict(tfidf_price_text, verbose=0)
    prediction_no_aspect = model_no_aspect.predict(tfidf_no_aspect_text, verbose=0)

    sentiment_attractions = np.argmax(prediction_attractions, axis=1)[0]
    sentiment_amenities = np.argmax(prediction_amenities, axis=1)[0]
    sentiment_access = np.argmax(prediction_access, axis=1)[0]
    sentiment_price = np.argmax(prediction_price, axis=1)[0]
    sentiment_no_aspect = np.argmax(prediction_no_aspect, axis=1)[0]

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    
    sentiment_attractions = label_map[sentiment_attractions]
    sentiment_amenities = label_map[sentiment_amenities]
    sentiment_access = label_map[sentiment_access]
    sentiment_price = label_map[sentiment_price]
    sentiment_no_aspect = label_map[sentiment_no_aspect]

    return {
        "attractions": sentiment_attractions,
        "amenities": sentiment_amenities,
        "access": sentiment_access,
        "price": sentiment_price,
        "no_aspect": sentiment_no_aspect
    }