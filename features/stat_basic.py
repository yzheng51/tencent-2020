"""
Extract basic statistical features from data
"""
import pandas as pd
from utils.timer import Timer
# -------------------------------------------------------------------------------------------------
TRAIN_FILE = 'data/train_preliminary/train.pkl'
TEST_FILE = 'data/test/test.pkl'

TRAIN_STAT_FEAT = 'data/train_feat/train_basic_stat_feat.pkl'
TEST_STAT_FEAT = 'data/test_feat/test_basic_stat_feat.pkl'

na_cols = [
    'product_id_count', 'product_id_nunique', 'industry_count', 'industry_nunique', 'duration_std'
]

dtype = {
    'creative_id_count': 'uint32',
    'creative_id_nunique': 'uint32',
    'ad_id_nunique': 'uint32',
    'advertiser_id_nunique': 'uint32',
    'product_category_nunique': 'uint32',
    'click_times_nunique': 'uint32',
    'click_times_max': 'uint8',
    'click_times_sum': 'uint32',
    'click_times_mean': 'float64',
    'click_times_std': 'float64',
    'time_nunique': 'uint32',
    'time_min': 'uint8',
    'time_max': 'uint8',
    'product_id_count': 'uint32',
    'product_id_nunique': 'uint32',
    'industry_count': 'uint32',
    'industry_nunique': 'uint32',
    'duration_nunique': 'uint32',
    'duration_min': 'uint8',
    'duration_max': 'uint8',
    'duration_mean': 'float64',
    'duration_median': 'float64',
    'duration_std': 'float64',
    'creative_id_count_bin_10': 'uint8',
    'creative_id_nunique_bin_10': 'uint8',
    'ad_id_nunique_bin_10': 'uint8',
    'advertiser_id_nunique_bin_10': 'uint8',
    'product_category_nunique_bin_10': 'uint8',
    'product_id_count_bin_10': 'uint8',
    'product_id_nunique_bin_10': 'uint8',
    'industry_count_bin_10': 'uint8',
    'industry_nunique_bin_10': 'uint8',
    'click_times_max_lt_1': 'uint8',
    'click_times_sum_bin_10': 'uint8',
    'click_times_mean_bin_2': 'uint8',
    'click_times_std_bin_2': 'uint8',
    'time_nunique_bin_10': 'uint8',
    'time_min_bin_4': 'uint8',
    'time_max_bin_2': 'uint8',
    'duration_nunique_bin_4': 'uint8',
    'duration_min_lt_1': 'uint8',
    'duration_max_bin_10': 'uint8',
    'duration_mean_bin_10': 'uint8',
    'duration_median_bin_4': 'uint8',
    'duration_std_bin_10': 'uint8'
}

timer = Timer()
# -------------------------------------------------------------------------------------------------
print('Loading train and test data...')
timer.start()
train = pd.read_pickle(TRAIN_FILE)
test = pd.read_pickle(TEST_FILE)

timer.stop()
# -------------------------------------------------------------------------------------------------
print('Generate basic statistical features')
timer.start()
train_stat_basic = pd.DataFrame()
test_stat_basic = pd.DataFrame()

# general
temp = train.groupby('user_id').agg({
    'creative_id': ['count', 'nunique'],
    'ad_id': ['nunique'],
    'advertiser_id': ['nunique'],
    'product_category': ['nunique'],
    'click_times': ['nunique', 'max', 'sum', 'mean', 'std'],
    'time': ['nunique', 'min', 'max']
})
temp.columns = ["_".join(x) for x in temp.columns.ravel()]
train_stat_basic = pd.concat([train_stat_basic, temp], axis=1)

temp = test.groupby('user_id').agg({
    'creative_id': ['count', 'nunique'],
    'ad_id': ['nunique'],
    'advertiser_id': ['nunique'],
    'product_category': ['nunique'],
    'click_times': ['nunique', 'max', 'sum', 'mean', 'std'],
    'time': ['nunique', 'min', 'max']
})
temp.columns = ["_".join(x) for x in temp.columns.ravel()]
test_stat_basic = pd.concat([test_stat_basic, temp], axis=1)

# product_id
temp = train.loc[train['product_id'] != '\\N'].groupby('user_id').agg({
    'product_id': ['count', 'nunique']
})
temp.columns = ["_".join(x) for x in temp.columns.ravel()]
train_stat_basic = pd.concat([train_stat_basic, temp], axis=1)

temp = test.loc[test['product_id'] != '\\N'].groupby('user_id').agg({
    'product_id': ['count', 'nunique']
})
temp.columns = ["_".join(x) for x in temp.columns.ravel()]
test_stat_basic = pd.concat([test_stat_basic, temp], axis=1)

# industry
temp = train.loc[train['industry'] != '\\N'].groupby('user_id').agg({
    'industry': ['count', 'nunique']
})
temp.columns = ["_".join(x) for x in temp.columns.ravel()]
train_stat_basic = pd.concat([train_stat_basic, temp], axis=1)

temp = test.loc[test['industry'] != '\\N'].groupby('user_id').agg({
    'industry': ['count', 'nunique']
})
temp.columns = ["_".join(x) for x in temp.columns.ravel()]
test_stat_basic = pd.concat([test_stat_basic, temp], axis=1)
timer.stop()
# -------------------------------------------------------------------------------------------------
print('Generate statistical features based on click date duration...')
timer.start()
# drop all columns except user_id and time
# since only time duration will be taken into consideration
# keep one click log record at each day
train = train.loc[:, ['user_id', 'time']].drop_duplicates().sort_values(['user_id', 'time'])
test = test.loc[:, ['user_id', 'time']].drop_duplicates().sort_values(['user_id', 'time'])

# create time duration statistical features
train['next_time'] = train.groupby('user_id')['time'].shift(-1)
temp = train.groupby('user_id').size()
train.loc[train['user_id'].isin(temp[temp == 1].index), 'next_time'] = 0
train = train.loc[train['next_time'].notna()]
train = train.astype({'next_time': 'uint8'})
train['duration'] = train['next_time'] - train['time']
temp = train.groupby('user_id').agg({'duration': ['nunique', 'min', 'max', 'mean', 'median', 'std']})
temp.columns = ["_".join(x) for x in temp.columns.ravel()]
train_stat_basic = pd.concat([train_stat_basic, temp], axis=1)

test['next_time'] = test.groupby('user_id')['time'].shift(-1)
temp = test.groupby('user_id').size()
test.loc[test['user_id'].isin(temp[temp == 1].index), 'next_time'] = 0
test = test.loc[test['next_time'].notna()]
test = test.astype({'next_time': 'uint8'})
test['duration'] = test['next_time'] - test['time']
temp = test.groupby('user_id').agg({'duration': ['nunique', 'min', 'max', 'mean', 'median', 'std']})
temp.columns = ["_".join(x) for x in temp.columns.ravel()]
test_stat_basic = pd.concat([test_stat_basic, temp], axis=1)

# fill nan values with zeros
train_stat_basic.loc[:, na_cols] = train_stat_basic.loc[:, na_cols].fillna(0)
test_stat_basic.loc[:, na_cols] = test_stat_basic.loc[:, na_cols].fillna(0)

timer.stop()
# -------------------------------------------------------------------------------------------------
print('Bucketing continuous features...')
timer.start()

train_stat_basic['creative_id_count_bin_10'] = pd.qcut(train_stat_basic['creative_id_count'], q=10).cat.codes
train_stat_basic['creative_id_nunique_bin_10'] = pd.qcut(train_stat_basic['creative_id_nunique'], q=10).cat.codes
train_stat_basic['ad_id_nunique_bin_10'] = pd.qcut(train_stat_basic['ad_id_nunique'], q=10).cat.codes
train_stat_basic['advertiser_id_nunique_bin_10'] = pd.qcut(train_stat_basic['advertiser_id_nunique'], q=10).cat.codes
train_stat_basic['product_category_nunique_bin_10'] = pd.qcut(train_stat_basic['product_category_nunique'], q=4).cat.codes
train_stat_basic['product_id_count_bin_10'] = pd.qcut(train_stat_basic['product_id_count'], q=10).cat.codes
train_stat_basic['product_id_nunique_bin_10'] = pd.qcut(train_stat_basic['product_id_nunique'], q=10).cat.codes
train_stat_basic['industry_count_bin_10'] = pd.qcut(train_stat_basic['industry_count'], q=10).cat.codes
train_stat_basic['industry_nunique_bin_10'] = pd.qcut(train_stat_basic['industry_nunique'], q=10).cat.codes
train_stat_basic['click_times_max_lt_1'] = train_stat_basic['click_times_max'].map(lambda s: 0 if s <= 1 else 1)
train_stat_basic['click_times_sum_bin_10'] = pd.qcut(train_stat_basic['click_times_sum'], q=10).cat.codes
train_stat_basic['click_times_mean_bin_2'] = pd.qcut(train_stat_basic['click_times_mean'], q=2).cat.codes
train_stat_basic['click_times_std_bin_2'] = pd.qcut(train_stat_basic['click_times_std'], q=2).cat.codes
train_stat_basic['time_nunique_bin_10'] = pd.qcut(train_stat_basic['time_nunique'], q=10).cat.codes
train_stat_basic['time_min_bin_4'] = pd.qcut(train_stat_basic['time_min'], q=4).cat.codes
train_stat_basic['time_max_bin_2'] = pd.qcut(train_stat_basic['time_max'], q=2).cat.codes
train_stat_basic['duration_nunique_bin_4'] = pd.qcut(train_stat_basic['duration_nunique'], q=4).cat.codes
train_stat_basic['duration_min_lt_1'] = train_stat_basic['duration_min'].map(lambda s: 0 if s <= 1 else 1)
train_stat_basic['duration_max_bin_10'] = pd.qcut(train_stat_basic['duration_max'], q=10).cat.codes
train_stat_basic['duration_mean_bin_10'] = pd.qcut(train_stat_basic['duration_mean'], q=10).cat.codes
train_stat_basic['duration_median_bin_4'] = pd.qcut(train_stat_basic['duration_median'], q=4).cat.codes
train_stat_basic['duration_std_bin_10'] = pd.qcut(train_stat_basic['duration_std'], q=10).cat.codes

test_stat_basic['creative_id_count_bin_10'] = pd.qcut(test_stat_basic['creative_id_count'], q=10).cat.codes
test_stat_basic['creative_id_nunique_bin_10'] = pd.qcut(test_stat_basic['creative_id_nunique'], q=10).cat.codes
test_stat_basic['ad_id_nunique_bin_10'] = pd.qcut(test_stat_basic['ad_id_nunique'], q=10).cat.codes
test_stat_basic['advertiser_id_nunique_bin_10'] = pd.qcut(test_stat_basic['advertiser_id_nunique'], q=10).cat.codes
test_stat_basic['product_category_nunique_bin_10'] = pd.qcut(test_stat_basic['product_category_nunique'], q=4).cat.codes
test_stat_basic['product_id_count_bin_10'] = pd.qcut(test_stat_basic['product_id_count'], q=10).cat.codes
test_stat_basic['product_id_nunique_bin_10'] = pd.qcut(test_stat_basic['product_id_nunique'], q=10).cat.codes
test_stat_basic['industry_count_bin_10'] = pd.qcut(test_stat_basic['industry_count'], q=10).cat.codes
test_stat_basic['industry_nunique_bin_10'] = pd.qcut(test_stat_basic['industry_nunique'], q=10).cat.codes
test_stat_basic['click_times_max_lt_1'] = test_stat_basic['click_times_max'].map(lambda s: 0 if s <= 1 else 1)
test_stat_basic['click_times_sum_bin_10'] = pd.qcut(test_stat_basic['click_times_sum'], q=10).cat.codes
test_stat_basic['click_times_mean_bin_2'] = pd.qcut(test_stat_basic['click_times_mean'], q=2).cat.codes
test_stat_basic['click_times_std_bin_2'] = pd.qcut(test_stat_basic['click_times_std'], q=2).cat.codes
test_stat_basic['time_nunique_bin_10'] = pd.qcut(test_stat_basic['time_nunique'], q=10).cat.codes
test_stat_basic['time_min_bin_4'] = pd.qcut(test_stat_basic['time_min'], q=4).cat.codes
test_stat_basic['time_max_bin_2'] = pd.qcut(test_stat_basic['time_max'], q=2).cat.codes
test_stat_basic['duration_nunique_bin_4'] = pd.qcut(test_stat_basic['duration_nunique'], q=4).cat.codes
test_stat_basic['duration_min_lt_1'] = test_stat_basic['duration_min'].map(lambda s: 0 if s <= 1 else 1)
test_stat_basic['duration_max_bin_10'] = pd.qcut(test_stat_basic['duration_max'], q=10).cat.codes
test_stat_basic['duration_mean_bin_10'] = pd.qcut(test_stat_basic['duration_mean'], q=10).cat.codes
test_stat_basic['duration_median_bin_4'] = pd.qcut(test_stat_basic['duration_median'], q=4).cat.codes
test_stat_basic['duration_std_bin_10'] = pd.qcut(test_stat_basic['duration_std'], q=10).cat.codes

timer.stop()
# -------------------------------------------------------------------------------------------------
# change data type to reduce memory usage and drop user_id index
train_stat_basic = train_stat_basic.reset_index(drop=True).astype(dtype)
test_stat_basic = test_stat_basic.reset_index(drop=True).astype(dtype)

print(f'\nSave feature to `{TRAIN_STAT_FEAT}` and `{TEST_STAT_FEAT}`')
train_stat_basic.to_pickle(TRAIN_STAT_FEAT)
test_stat_basic.to_pickle(TEST_STAT_FEAT)
