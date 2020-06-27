"""
Generate id sequence for each user
"""
import pandas as pd
from utils.timer import Timer
# -------------------------------------------------------------------------------------------------
TRAIN_FILE = 'data/train_preliminary/train.pkl'
TEST_FILE = 'data/test/test.pkl'

TRAIN_SEQ_FILE = 'data/train_preliminary/train_seq.pkl'
TEST_SEQ_FILE = 'data/test/test_seq.pkl'

SEQ_TO_GENERATE = [
    'creative_id',
    'ad_id',
    'product_id',
    'advertiser_id',
    'product_category',
    'industry',
    'time',
    'click_times'
]
print(f'To generate features: {SEQ_TO_GENERATE}')

timer = Timer()
# -------------------------------------------------------------------------------------------------
print('Loading train and test data...')
timer.start()
train = pd.read_pickle(TRAIN_FILE)
test = pd.read_pickle(TEST_FILE)

train = train.astype({'time': 'str', 'click_times': 'str'})
test = test.astype({'time': 'str', 'click_times': 'str'})
timer.stop()
# -------------------------------------------------------------------------------------------------
print('Make rank column base on time')
timer.start()
train['rn'] = train \
    .sort_values('time', ascending=True) \
    .groupby('user_id') \
    .cumcount() + 1
train.sort_values(['user_id', 'rn'], inplace=True)

test['rn'] = test \
    .sort_values('time', ascending=True) \
    .groupby('user_id') \
    .cumcount() + 1
test.sort_values(['user_id', 'rn'], inplace=True)
timer.stop()
# -------------------------------------------------------------------------------------------------
print('Create id sequence features for each user...')
timer.start()
train = train.groupby('user_id')[SEQ_TO_GENERATE] \
    .agg(lambda group: [s for s in group if s != '\\N']) \
    .reset_index(level=0)

test = test.groupby('user_id')[SEQ_TO_GENERATE] \
    .agg(lambda group: [s for s in group if s != '\\N']) \
    .reset_index(level=0)
timer.stop()

# save ad_id temp file
print('Save user based table...')
timer.start()
train.to_pickle(TRAIN_SEQ_FILE)
test.to_pickle(TEST_SEQ_FILE)
timer.stop()
