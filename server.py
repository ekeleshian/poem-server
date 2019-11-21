
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
from fastai.text import *

app = Flask(__name__)
CORS(app, support_credentials=True)
PATH = Path('/Users/ekeleshian/poem-server/')


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        req=request.data
        print(str(request))
        return str(req)
    else:
        return "Listening..."


@app.route('/classify/<words>', methods=['GET'])
@cross_origin(supports_credentials=True)
def classify(words):
    print(request.get_data())
    response = Response()
    response.headers['Content-Security-Policy'] = "connect-src '*'"
    response.headers.add("Access-Control-Allow-Origin", "*")
    words = str(words)
    words = words.replace('_', ' ')
    res = learn.predict(words, 100, temperature=1.5, min_p=0.001, no_unk=True)
    res = res.replace('xxbos', '')
    idx = res[::-1].find('.')
    if idx == -1:
        response.set_data(res)
        return response
    final = res[:len(res) - idx]
    response.set_data(final)
    return response


if __name__ == "__main__":

    data_lm = load_data(PATH, 'data_save.pkl', bs=10)

    learn = language_model_learner(data_lm, arch=AWD_LSTM, pretrained=True, drop_mult=0.3)

    learn.load('poems7')

    app.run(host='0.0.0.0', port=5005)