from keras.models import load_model
import tensorflow as tf
import os
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template,url_for
import pickle
import re
import nltk#natural language tool kit
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
with open(r'countvectorizer.pkl','rb') as file:
    cv=pickle.load(file)
cla=load_model('phone_review.h5',compile=False) 
cla.compile(optimizer='adam',loss='binary_crossentropy')
app = Flask(__name__)

@app.route('/')
def home():    
    return render_template('home.html')

@app.route('/select',methods=['POST'])
def select():
    return render_template('base.html')


@app.route('/text')
def text():
    return render_template('text.html')


@app.route('/project')
def reviewtext():
    return render_template('reviewtext.html')

@app.route('/page')
def page():
    return render_template('base.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        img_url = url_for('static',filename = 'style/3.jpg')
        return render_template('base.html',url=img_url)
    if request.method == 'POST':
        topic = request.form['ms']
        review=re.sub("[^a-zA-Z]"," ",topic)
        review=review.lower()
        review=review.split()    
        review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        topic=" ".join(review)
        print("I m" +topic)
        topic=cv.transform([topic])
        print("\n"+str(topic.shape)+"\n")
        with graph.as_default():
            y_pred = cla.predict(topic)
            print("pred is "+str(y_pred))
        if(y_pred>0.5):
            img_url = url_for('static',filename = 'style/1.jpg')
            topic = "happy"
        elif(y_pred<0.5):
            img_url = url_for('static',filename = 'style/2.jpg')
            topic = "sad"
        return render_template('base.html',ypred = topic)
       
if __name__ == '__main__':
    app.run(debug = False, threaded = True)

