"""
Merge user click log and ad ids into one table
"""
import pandas as pd
from utils.timer import Timer
# -------------------------------------------------------------------------------------------------
TRAIN_CLICK_LOG_FILE = 'data/train_preliminary/click_log.csv'
TRAIN_AD_FILE = 'data/train_preliminary/ad.csv'
TEST_CLICK_LOG_FILE = 'data/test/click_log.csv'
TEST_AD_FILE = 'data/test/ad.csv'

TRAIN_FILE = 'data/train_preliminary/train.pkl'
TEST_FILE = 'data/test/test.pkl'

dtype = {
    'user_id': 'int32',
    'creative_id': 'str',
    'ad_id': 'str',
    'product_id': 'str',
    'product_category': 'str',
    'advertiser_id': 'str',
    'industry': 'str',
    'time': 'uint8',
    'click_times': 'uint8',
}

timer = Timer()
# -------------------------------------------------------------------------------------------------
print('Loading train and test data...')
timer.start()
train_click_log = pd.read_csv(TRAIN_CLICK_LOG_FILE, dtype=dtype)
train_ad = pd.read_csv(TRAIN_AD_FILE, dtype=dtype)

test_click_log = pd.read_csv(TEST_CLICK_LOG_FILE, dtype=dtype)
test_ad = pd.read_csv(TEST_AD_FILE, dtype=dtype)
timer.stop()
# -------------------------------------------------------------------------------------------------
print('Merging user log and ad id...')
timer.start()
train = pd.merge(train_click_log, train_ad, how='left', on='creative_id')
test = pd.merge(test_click_log, test_ad, how='left', on='creative_id')

train.to_pickle(TRAIN_FILE)
test.to_pickle(TEST_FILE)
timer.stop()
