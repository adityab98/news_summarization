import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.corpus import stopwords
from newspaper import Article
from GoogleNews import GoogleNews
import csv
import pickle

def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

stop_words = stopwords.words('english')
googlenews=GoogleNews()
f = open("embeddings.pkl","rb")
word_embeddings=pickle.load(f)
max=3 #change this to change the number of articles that the program will go through for the summar
count=0 

print("---------------")
print("Enter the topic:")
topic=input()
print("---------------")
googlenews.search(topic)

for link in googlenews.get_links():
    if(count==max):
        break
    print(link)
    sentences=[]
    article=Article(link)
    article.download()
    article.parse()
    article.nlp()
    with open('article.csv','w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(["article_text","source"])
        writer.writerow([article.text,link])
    df=pd.read_csv("article.csv")
    for s in df["article_text"]:
        sentences.append(sent_tokenize(s))
    sentences = [y for x in sentences for y in x]
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    sentence_vectors = []
    for i in clean_sentences:
      if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
      else:
        v = np.zeros((100,))
      sentence_vectors.append(v)
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
      for j in range(len(sentences)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    if(len(ranked_sentences)<3):
        for i in range(len(ranked_sentences)):
            print(ranked_sentences[i][1])
    else: #if more sentences are required, simply add another line with print(ranked_sentences[n][1]). However, please note that some articles and webpages do not play well with the libraries used in this program. The default of the top 3 sentences is suggested
        print(ranked_sentences[0][1])
        print(ranked_sentences[1][1])
        print(ranked_sentences[2][1])
    print("---------------")
    count+=1
