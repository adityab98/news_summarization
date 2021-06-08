import nltk
import pickle
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
f=open("embeddings.pkl","wb")
pickle.dump(word_embeddings, f)
