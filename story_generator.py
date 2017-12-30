
from keras.models import model_from_json
import numpy as np
print('keras imported')

text=open('train.txt').read().lower().replace('\xa0',' ')
chars = sorted(list(set(text)))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen=100
start_index=200
diversity=0.1
char_count=0

json_string=open('model-final.json','r').read()
model = model_from_json(json_string)
model.load_weights('weights-final.h5')
print('Model loaded')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    np.seterr(all='raise')
    try:
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    except:
        return np.argmax(preds)

generated = ''
seed = ''
#Create the seed to start generating text from
for i in range (maxlen-4):
    seed+=np.random.choice(chars)
#End the seed with " and" so that the next generated word actually makes sense. 
seed+=' and'
#generated += seed
print('----- Generating with seed: "' + seed + '"')

for chapter_no in range(20):
    print('chapter no:',chapter_no+1)
    char_count=0
    generated+='chapter '+str(chapter_no+1)+'\n'
    while True:
        print('char count:',char_count)
        if next_char=='.' and np.random.choice([True,False],p=[0.3,0.7]):
            #after each sentence, a 30% chance of changing paragraphs regardless of what the model outputs
            next_char='\n'
        else:
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(seed):
                x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds,diversity)
            next_char = indices_char[next_index]
        generated += next_char
        char_count+=1
        seed = seed[1:] + next_char
        if char_count>10000 and next_char=='\n' and np.random.choice([True,False]):
            #after 10000 characters, after each paragraph, a 50% chance of changing chapters
            generated+='\n'
            break

with open("story.txt", "w") as story:
    story.write(generated)

