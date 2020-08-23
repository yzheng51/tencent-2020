import gc
import pickle
import gensim
import numpy as np
import pandas as pd
from utils.timer import Timer
from utils.callback import MyLossCalculator
# -------------------------------------------------------------------------------------------------
# word2vec hyper parameters
SIZE = 128
WINDOW = 50
MIN_COUNT = 2

seed = 2020

TRAIN_SEQ_FILE = 'data/train_preliminary/train_seq.pkl'
TEST_SEQ_FILE = 'data/test/test_seq.pkl'
CORPUS_FEAT = 'industry'

CORPUS_FILE = f'data/corpus/{CORPUS_FEAT}.pkl'
EMBED_MATRIX_FILE = f'data/embed/{CORPUS_FEAT}.npy'
EMBED_MODEL_FILE = f'data/w2v_model/w2v_{CORPUS_FEAT}.model'

timer = Timer()
# -------------------------------------------------------------------------------------------------
# read data
print('Loading train, test data and concatenate them...')
timer.start()
train = pd.read_pickle(TRAIN_SEQ_FILE)
test = pd.read_pickle(TEST_SEQ_FILE)
corpus = pd.concat([train[CORPUS_FEAT], test[CORPUS_FEAT]]).tolist()

timer.stop()

# call garbage collection to release some memory
del train, test
gc.collect()
# -------------------------------------------------------------------------------------------------
# train word2vec
print(f'Train word2vec on {CORPUS_FEAT} and save model...')
timer.start()
model = gensim.models.Word2Vec(
    corpus, size=SIZE, window=WINDOW, min_count=MIN_COUNT, iter=10, sg=1, hs=1, workers=10,
    seed=seed, compute_loss=True, callbacks=[MyLossCalculator()]
)
timer.stop()

print('Convert words in corpus to indexes...')
timer.start()
# make word to index map
word_to_idx = {}
for idx, word in enumerate(model.wv.index2word, start=1):
    word_to_idx[word] = idx

# convert corpuse to index based on trained model
indexed_corpus = list()
for sent in corpus:
    temp = [word_to_idx[word] for word in sent if word in word_to_idx]
    indexed_corpus.append(temp)

timer.stop()

# save indexed corpus and embedding matrix
print(f'Save corpus file to `{CORPUS_FILE}`')
with open(CORPUS_FILE, 'wb') as f:
    pickle.dump(indexed_corpus, f)

# add zeros for padding
print(f'Save embedding matrix to `{EMBED_MATRIX_FILE}`')
embed_matrix = np.vstack((np.zeros(SIZE), model.wv.vectors))
np.save(EMBED_MATRIX_FILE, embed_matrix)

print(f'Save model file to `{EMBED_MODEL_FILE}`')
model.save(EMBED_MODEL_FILE)
