import pickle5 as pickle

with open('model/ixtoword.pkl', 'rb') as handle:
    ixtoword = pickle.load(handle)
with open('model/wordtoidx.pkl', 'rb') as handle:
    wordtoix = pickle.load(handle)
