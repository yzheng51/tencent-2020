import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils.stacking import kfold_target_mean
from utils.timer import Timer
# -------------------------------------------------------------------------------------------------
seed = 2020
np.random.seed(seed)

TRAIN_FILE = 'data/train_preliminary/train.pkl'
TEST_FILE = 'data/test/test.pkl'
LABEL_FILE = 'data/train_preliminary/user.csv'

TRAIN_GENDER_ENCODED_FEAT = 'data/train_feat/train_target_encode_gender_feat.pkl'
TEST_GENDER_ENCODED_FEAT = 'data/test_feat/test_target_encode_gender_feat.pkl'
TRAIN_AGE_ENCODED_FEAT = 'data/train_feat/train_target_encode_age_feat.pkl'
TEST_AGE_ENCODED_FEAT = 'data/test_feat/test_target_encode_age_feat.pkl'

dtype = {'user_id': 'int32', 'age': 'uint8', 'gender': 'uint8'}

FEAT_TO_GENERATE = [
    'creative_id',
    'ad_id',
    'product_id',
    'advertiser_id',
    'product_category',
    'industry',
]

timer = Timer()
# -------------------------------------------------------------------------------------------------
print('Loading train and test data...')
timer.start()
train = pd.read_pickle(TRAIN_FILE)
test = pd.read_pickle(TEST_FILE)
user = pd.read_csv(LABEL_FILE, dtype=dtype)

timer.stop()
# -------------------------------------------------------------------------------------------------
train = train.merge(user, how='left', on='user_id')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
# -------------------------------------------------------------------------------------------------
print('Generate target encoding features for gender')
train_target_encode = train['user_id']
test_target_encode = test['user_id']

target_sr = train['gender'] - 1

for cat in FEAT_TO_GENERATE:
    timer.start()
    stack_train, stack_test = kfold_target_mean(kfold, train[cat], target_sr, test[cat])
    timer.stop()
    temp = pd.Series(stack_train, name=f'{cat}_target_mean')
    train_target_encode = pd.concat([train_target_encode, temp], axis=1)
    temp = pd.Series(stack_test, name=f'{cat}_target_mean')
    test_target_encode = pd.concat([test_target_encode, temp], axis=1)

train_target_encode = train_target_encode.groupby('user_id').mean().reset_index(drop=True)
test_target_encode = test_target_encode.groupby('user_id').mean().reset_index(drop=True)

print(f'\nSave feature to `{TRAIN_GENDER_ENCODED_FEAT}` and `{TEST_GENDER_ENCODED_FEAT}`')
train_target_encode.to_pickle(TRAIN_GENDER_ENCODED_FEAT)
test_target_encode.to_pickle(TEST_GENDER_ENCODED_FEAT)
