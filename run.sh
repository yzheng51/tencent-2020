#!/usr/bin/bash

mkdir -p data/train_feat/
mkdir -p data/test_feat/
mkdir -p data/corpus/
mkdir -p data/embed/
mkdir -p data/w2v_model/
# merge user click log and ad ids into one table
python -m preprocessing.concat

# create id sequence for each user
python -m preprocessing.sequence

# create basic statistical features
python -m features.stat_basic

# create target encoding features
python -m features.target_encoding

# stacking tfidf features
python -m features.tfidf_creative_id
python -m features.tfidf_ad_id
python -m features.tfidf_advertiser_id
python -m features.tfidf_product_id
python -m features.tfidf_industry
python -m features.tfidf_product_category

# create embedding for each id with word2vec
python -m embeddings.word2vec_creative_id
python -m embeddings.word2vec_ad_id
python -m embeddings.word2vec_advertiser_id
python -m embeddings.word2vec_product_id
python -m embeddings.word2vec_industry
python -m embeddings.word2vec_product_category

# neural network example
python -m nn.tf_lstm
