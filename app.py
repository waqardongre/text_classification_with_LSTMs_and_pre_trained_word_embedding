from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
import tensorflow as tf
import pickle
from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
import numpy as np


app = FastAPI()

@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <html>
        <body>
        <form method="post" class="form">
        <p> Enter your review below: </p>
        <input maxlength="1000" name="text" type="text" value="this movie is awesome" />
        <input type="submit" value="Predict"/>
        <p>Sample inputs:</p>
        <p>this movie is awesome</p>
        <p>very slow movie</p>
        <p>Deep learning model: LSTMs with 79'%' accuracy on 20'%' data with pre-trained word embedding 'glove.6B.zip'</p>
        <p>By: Waqar Dongre</p>
        <p>Email: waqardongre@gmail.com</p>
        </html>
        </body>'''

@app.post('/predict')
def predict(text:str = Form(...)):
    
    loaded_model = tf.keras.models.load_model('LSTM_h5_model_4_acc_80_perc.h5') #load the saved model 

    from_disk = pickle.load(open("vectorizer.pkl", "rb"))
    from_disk['config']["vocabulary"] = "imdb.vocab" # set you vocabulary file path

    vectorizer = TextVectorization.from_config(from_disk['config'])
    vectorizer.set_weights(from_disk['weights'])
    
    string_input = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_input)
    preds = loaded_model(x)
    end_to_end_model = keras.Model(string_input, preds)

    probabilities = end_to_end_model.predict(
        [[text]]
    )
    class_names = ['Positive', 'Negative'] # as defind in the model 0 for positive and 1 for negative
    t_sentiment = class_names[int(0.5 > probabilities[0][0])]

    return { #return the dictionary for endpoint
        "your_review": text,
        "predicted_sentiment": t_sentiment
    }
    #return '''Your input: \''''+ str(text) + '''\' and your review is predicted as: ''' + str(t_sentiment)