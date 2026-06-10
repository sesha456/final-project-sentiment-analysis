from flask import Flask, request, render_template
import numpy as np
import pickle
from regex_extract import user_data
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import os

# Download required NLTK data on startup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
model = pickle.load(open("final.pkl","rb"))
model1 = pickle.load(open("final_vector.pkl","rb"))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods = ['post','get'])
def predict():
    print(request.form)
    form_feature = request.form["review"]
    print(form_feature)
    data = user_data(form_feature)
    fea_usr_data = model1.transform(data).toarray()
    pred = model.predict(fea_usr_data)
    print(type(pred))
    print("list of pred :",pred)
    if pred[-1]==1:
        return render_template("index.html",prediction_text="It is a Positive Sentiment.")
    else:
        return render_template("index.html",prediction_text="It is a Negative Sentiment.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
