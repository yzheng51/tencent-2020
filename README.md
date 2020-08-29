# Tencent algorithm competition 2020

Join click log and ad features

```sh
python -m preprocessing.concat
```

Create clicked ad sequence for each user

```sh
python -m preprocessing.sequence
```

Statistical features

```sh
python -m features.stat_basic
```

Target Encoding

```sh
python -m features.target_encoding
```

TF-IDF features for each id

```sh
python -m features.tfidf_creative_id
python -m features.tfidf_ad_id
python -m features.tfidf_advertiser_id
python -m features.tfidf_product_id
python -m features.tfidf_industry
python -m features.tfidf_product_category
```

Embedding

```sh
python -m embeddings.word2vec_creative_id
python -m embeddings.word2vec_ad_id
python -m embeddings.word2vec_advertiser_id
python -m embeddings.word2vec_product_id
python -m embeddings.word2vec_industry
python -m embeddings.word2vec_product_category
```

Neural network example

```sh
python -m nn.tf_lstm
```
