"""
Extract term frequency features or tfidf features from data and fit them with simple model
"""
import gc
import random
import scipy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from utils.timer import Timer
from utils.stacking import kfold_stack_binary, kfold_stack_multi
# -------------------------------------------------------------------------------------------------
# read data

seed = 2020
random.seed(seed)
np.random.seed(seed)

MIN_COUNT = 2
TARGET_FEAT = 'creative_id'
N_CLS = 10

TRAIN_SEQ_FILE = 'data/train_preliminary/train_seq.pkl'
TEST_SEQ_FILE = 'data/test/test_seq.pkl'
LABEL_FILE = 'data/train_preliminary/user.csv'

TRAIN_GENDER_FEAT = f'data/train_feat/train_tfidf_gender_feat_{TARGET_FEAT}.pkl'
TEST_GENDER_FEAT = f'data/test_feat/test_tfidf_gender_feat_{TARGET_FEAT}.pkl'
TRAIN_AGE_FEAT = f'data/train_feat/train_tfidf_age_feat_{TARGET_FEAT}.pkl'
TEST_AGE_FEAT = f'data/test_feat/test_tfidf_age_feat_{TARGET_FEAT}.pkl'

dtype = {'user_id': 'int32', 'age': 'uint8', 'gender': 'uint8'}

timer = Timer()
# -------------------------------------------------------------------------------------------------
print('Loading data and preprocessing...')
timer.start()
train = pd.read_pickle(TRAIN_SEQ_FILE)
test = pd.read_pickle(TEST_SEQ_FILE)
user = pd.read_csv(LABEL_FILE)

label_gender = user.gender.values - 1
label_age = user.age.values - 1

# concatenate train and test into one dataframe
concated_data = pd.concat([train[TARGET_FEAT], test[TARGET_FEAT]]) \
    .reset_index(level=0, drop=True) \
    .tolist()
timer.stop()
# -------------------------------------------------------------------------------------------------
print(f'Generate vector space model for {TARGET_FEAT}...')
timer.start()
vectorizer = CountVectorizer(
    min_df=MIN_COUNT,
    tokenizer=lambda s: s,
    preprocessor=lambda s: s,
    token_pattern=None,
)

tf_transformer = TfidfTransformer(use_idf=False)
tfidf_transformer = TfidfTransformer(use_idf=True)

tf_csr = vectorizer.fit_transform(concated_data)
tfidf_csr = tfidf_transformer.fit_transform(tf_csr)
tf_csr = tf_transformer.fit_transform(tf_csr)

# merge two features
concated_data = scipy.sparse.csr_matrix(scipy.sparse.hstack([tf_csr, tfidf_csr]))
print(f'Dimension of the sparse matrix is {concated_data.shape}.')
timer.stop()
# -------------------------------------------------------------------------------------------------
# specify train, test and create spliter
x_train = concated_data[:len(train)]
x_test = concated_data[len(train):]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# call garbage collection to release some memory
del train, test, user, tf_csr, tfidf_csr
gc.collect()
# -------------------------------------------------------------------------------------------------

print('-' * 100)
print(f'Gender prediction with {TARGET_FEAT}\n')
models = dict(
    lr=LogisticRegression(random_state=seed, solver='sag'),
    svm=LinearSVC(random_state=seed, C=0.1),
    pac=PassiveAggressiveClassifier(random_state=seed, C=0.01),
    ridge=RidgeClassifier(random_state=seed, alpha=5),
    sgd=SGDClassifier(random_state=seed, loss='log', alpha=1e-6),
    bnb=BernoulliNB(),
    mnb=MultinomialNB(alpha=0.1),
)

# specify target label
y_train = label_gender

# define features
train_feat_gender = pd.DataFrame()
test_feat_gender = pd.DataFrame()

for name, model in models.items():
    timer.start()
    stack_train, stack_test = kfold_stack_binary(kfold, model, x_train, y_train, x_test)
    timer.stop()
    train_feat_gender[f'tfidf_feat_{TARGET_FEAT}_{name}'] = stack_train
    test_feat_gender[f'tfidf_feat_{TARGET_FEAT}_{name}'] = stack_test

print(f'\nSave feature to `{TRAIN_GENDER_FEAT}` and `{TEST_GENDER_FEAT}`')
train_feat_gender.to_pickle(TRAIN_GENDER_FEAT)
test_feat_gender.to_pickle(TEST_GENDER_FEAT)
# -------------------------------------------------------------------------------------------------
print('-' * 100)
print(f'Age prediction with {TARGET_FEAT}\n')
models = dict(
    lr=LogisticRegression(random_state=seed, solver='sag', n_jobs=N_CLS),
    svm=LinearSVC(random_state=seed, C=0.1),
    pac=PassiveAggressiveClassifier(random_state=seed, loss='squared_hinge', C=0.001, n_jobs=N_CLS),
    ridge=RidgeClassifier(random_state=seed, alpha=5),
    sgd=SGDClassifier(random_state=seed, loss='log', alpha=1e-6, n_jobs=N_CLS),
    bnb=BernoulliNB(alpha=0.1),
    mnb=MultinomialNB(alpha=0.1),
)

# specify target label
y_train = label_age

# define features
train_feat_age = pd.DataFrame()
test_feat_age = pd.DataFrame()

for name, model in models.items():
    timer.start()
    stack_train, stack_test = kfold_stack_multi(N_CLS, kfold, model, x_train, y_train, x_test)
    timer.stop()
    columns = [f'tfidf_feat_{TARGET_FEAT}_{name}_{i + 1}' for i in range(N_CLS)]
    train_feat_age = pd.concat([train_feat_age, pd.DataFrame(stack_train, columns=columns)], axis=1)
    test_feat_age = pd.concat([test_feat_age, pd.DataFrame(stack_test, columns=columns)], axis=1)

print(f'\nSave feature to `{TRAIN_AGE_FEAT}` and `{TEST_AGE_FEAT}`')
train_feat_age.to_pickle(TRAIN_AGE_FEAT)
test_feat_age.to_pickle(TEST_AGE_FEAT)
