import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('all')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



app = Flask(__name__)
model = pickle.load(open('model_RFC.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
    text = [str(x) for x in request.form.values()]
    text = ' '.join(text) #merge the subject and text to a single string 
    text = text.lower() # to convert the text to lowercase
    text = re.sub('[^a-zA-Z]', ' ',text) # to remove number and special characters 
    text = text.split()  #to tokenize the text
    wnl = WordNetLemmatizer()
    text = [wnl.lemmatize(word) for word in text if not word in stop_words] #to lemmatize and remove stopwords
    text = [word for word in text if len(word) >=3] #remove 3 or less characters; only keep words of length greater than 3
    text = ' '.join(text)
    if len(text) > 1000:
        vectorizer = TfidfVectorizer(max_features=1000, lowercase=False, ngram_range=(1,3))
        text_vec = vectorizer.fit_transform([text]).toarray()
        prediction = model.predict(text_vec)
        if prediction[0] == 0:
            output = 'False'
        else:
            output = 'True'

        return render_template('index.html', prediction_text='Prediction: {}'.format(output))
    else:
        return render_template('index.html', prediction_text='Prediction: {}'.format(f'Number of keyword features ({len(text)}) is less than 1000'))

if __name__ == "__main__":
    app.run(debug=True)
    