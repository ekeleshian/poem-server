from fastai.text import *

PATH = Path('/Users/ekeleshian/poem-server/')

data_lm = load_data(PATH, 'data_save.pkl', bs=10)

learn = language_model_learner(data_lm, arch=AWD_LSTM, pretrained=True, drop_mult=0.3)

learn.load('poem5')


res = learn.predict("My love is sweet and so ", 80, temperature=1.1, min_p=0.001)

print(res)