import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle


with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output= pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
for intents in data ["intents"]:
    for pattern in intents["patterns"]: #Here we are stemming by getting the root of the word
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intents["tag"])

    if intents["tag"] not in labels:
        labels.append(intents["tag"])
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []
out_empty = [0 for_in range(len(labels))]
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc] 

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
output_row = out_empty[:]
output_row[labels.index(docs_y[x])] = 1

training.append(bag)
output.append(output_row)



training = numpy.array(training)
output = numpy.array(output)
with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output),f) 

tensorflow.reset_default_graph()

net = tflearn.input_data(shape =[None, len(training[0])])
net = tflearn.fulyy_connected(net, 8)
net = tflearn.fulyy_connected(net, 8)
net = tflearn.fulyy_connected(net, len(output[0]), activation = "softmax")
net.tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("tflearn")
except:
    model.fit(training, output, n_epoch=100, batch_size=8,  show_metric=True)
    model.save("model.tflearn")
def bag_of_words(s, words):
    bag = []

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, e in enumerate(words):
             if w == se:
                bag[i].append(1)
    return numpy.array(bag)















