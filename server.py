import re
import string
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
from fastai.text import *
from pdb import set_trace
app = Flask(__name__)
CORS(app, support_credentials=True)
PATH = Path('/Users/ekeleshian/poem-server/')

@app.route('/classify/<words>/<word_count>', methods=['GET'])
@cross_origin(supports_credentials=True)
def classify(words, word_count):
    response = Response()
    words = str(words)
    word_count = int(word_count)
    words = words.replace('_', ' ')
    res = learn.predict(words, word_count, temperature=1.5, min_p=0.001, no_unk=True)
    res = res.replace('xxbos', '')
    idx = res[::-1].find('.')
    if idx == -1:
        response.set_data(res)
        return response
    final = res[:len(res) - idx]
    text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', final)
    response.set_data(text)
    return response


if __name__ == "__main__":
    data_lm = load_data(PATH, 'data_save.pkl', bs=10)
    learn = language_model_learner(data_lm, arch=AWD_LSTM, pretrained=True, drop_mult=0.3)
    learn.load('poems7')
    app.run(host='0.0.0.0', port=5005, debug=True)