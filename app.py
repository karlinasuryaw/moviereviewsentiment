

from flask import Flask, request, render_template
import os
from flask import Flask, flash, request, redirect, url_for
import nltk
import re
import string

from werkzeug.utils import secure_filename
import numpy as np
import pickle
import sys
import logging

from nltk.tokenize import word_tokenize
import requests
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
import sqlite3


app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model_adam = load_model(
    'model_optimal_adam.h5'
)
model_rms = load_model(
    'model_optimal_rms.h5'
)


def cleaner(data):
    return(data.translate(str.maketrans('', '', string.punctuation)))

# dataframe.title = dataframe.title.apply(lambda x: cleaner(dataframe.title))
# dataframe.review = dataframe.review.apply(lambda x: cleaner(dataframe.review))


stwrds = stopwords.words('english')


def stopword(data):
    return(' '.join([w for w in data.split() if w not in stwrds]))


def get_opt(default):
    opt_list = ['Adam', 'RMSProp']
    default_opt = 'Adam'
    if default in opt_list:
        default_opt = default
    opt_list.remove(default_opt)
    return (opt_list, default_opt)


app = Flask(__name__)


@app.route('/search', methods=['POST', 'GET'])
def find_title():
    try:
        if request.method == 'POST':
            title_list = []
            pred_list = []
            title_movie = request.form.get('search')
            conn = sqlite3.connect('datatugasakhirfix.db')
            c = conn.cursor()
            title_movie = '%'+title_movie+'%'
            query_code = 'select * from ulasan where title like "{}"'.format(
                title_movie)
            title_list = c.execute(query_code).fetchall()
            # list_ulasan = title_list[0][1]
            conn.commit()
            conn.close()

            # title = [i[0] for i in title_list]
            review = [i[1] for i in title_list]

            tokenizer.fit_on_texts(review)
            sequence_text = tokenizer.texts_to_sequences(review)
            tokenization_rev = pad_sequences(sequence_text)
            pred = model_rms.predict(tokenization_rev)

            pos = 'Positif'
            neg = 'Negatif'
            label = []
            for i in pred:
                if i[0] > i[1]:
                    label.append(neg)
                else:
                    label.append(pos)

            pred_label = list(zip(pred, label))

            final_rest = list(zip(title_list, pred_label))

            return render_template('search.html', title_list=final_rest, hide=False)
    except:
        print("ada yg error")
        return render_template('search.html', hide=True, notfound=True)
    return render_template('search.html', hide=True)


@ app.route('/', methods=['POST', 'GET'])
def pred_sent():
    opt_list, default_opt = get_opt(request.form.get('select_opt'))
    hide = True
    notfound = False
    try:
        if request.method == 'POST':
            pred = []
            title = request.form['title']
            title_raw = request.form['title']
            print("isi judul ", title)
            review = request.form['review']
            review_raw = request.form['review']

            dataframe = pd.DataFrame(
                [[title, review]], columns=['title', 'review'])

            dataframe.review = dataframe.review.apply(lambda x: x.lower())
            dataframe.review = dataframe.review.apply(lambda x: cleaner(x))
            dataframe.review = dataframe.review.apply(lambda x: stopword(x))
            dataframe['review'] = dataframe['review'].str.replace(r'\d+', '')
            X = dataframe['review']
            tokenizer.fit_on_texts(X)
            sequence_text = tokenizer.texts_to_sequences(X)
            X = pad_sequences(sequence_text)

            opt_list, default_opt = get_opt(request.form.get('select_opt'))

            if default_opt == 'Adam':
                pred = model_adam.predict(X)
                print('Optimizer Selected : {}'.format(default_opt))
            elif default_opt == 'RMSProp':
                pred = model_rms.predict(X)
                print('Optimizer Selected : {}'.format(default_opt))

            nilai_besar = max(pred[0])

            max_val = 0
            idx_val = 0
            for i in range(len(pred[0])):
                if pred[0][i] > 0.5:
                    max_val = pred[0][i]
                    idx_max = i
                    if idx_max == 1:
                        tag_pred = 'Positif'
                        score = 1
                    elif idx_max == 0:
                        tag_pred = 'Negatif'
                        score = 0

            if request.form.get('saveornot') == 'save':
                conn = sqlite3.connect('datatugasakhirfix.db')
                c = conn.cursor()
                query_code = 'insert into ulasan (title, review, score) values ("{}", "{}", "{}")'.format(
                    title_raw, review_raw, score)
                c.execute(query_code)
                conn.commit()
                conn.close()

                print("Sukses disimpan ke database")
            return render_template('form.html', max_acc=max_val,
                                   tag_pred=tag_pred,
                                   review_clean=dataframe['review'].values, title=title_raw,
                                   review_raw=review_raw,
                                   pred_res=pred[0], opt=default_opt, hide=False,
                                   opt_list=opt_list, default_opt=default_opt)
    except:
        return render_template('form.html', opt_list=opt_list, default_opt=default_opt, notfound=True, hide=True)

    return render_template('form.html', opt_list=opt_list, default_opt=default_opt, hide=True)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.2", port=5002, threaded=True)
