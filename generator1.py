
from keras.models import model_from_json
import numpy as np
print('keras imported')

text=open('train.txt').read().lower().replace('\xa0',' ')
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen=100
start_index=200
diversity=0.05

json_string=open('model-final.json','r').read()
model = model_from_json(json_string)
model.load_weights('weights-final.h5')
print('Model loaded')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

generated = ''
seed = ''
for i in range (maxlen-4):
    seed+=np.random.choice(chars)
seed+=' and'
#generated += seed
print('----- Generating with seed: "' + seed + '"')

for i in range(400):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(seed):
        x_pred[0, t, char_indices[char]] = 1.
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds,diversity)
    next_char = indices_char[next_index]
    generated += next_char
    seed = seed[1:] + next_char
print(generated)