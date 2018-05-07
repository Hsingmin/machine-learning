# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		numpy 1.13.1
#		pandas 0.20.3
#		sklearn 0.19.0
#
# -*- author: Hsingmin Lee
#
# cvr_ffm.py -- Create FFM model to get CVR prediction .
#
import os
import sys
import codecs
import numpy as np
import pandas as pd
import csv
import data.dataset as dt
import model.gbmc as gc

# Train slices store path
DATASET_DIR = 'd:/engineering-data/Ali-IJCAI-18-data'

TRAIN_DATASET_RAW = 'round1_ijcai_18_train_20180301.txt'

TEST_DATASET_RAW = 'round1_ijcai_18_test_a_20180301.txt'

# TRAIN_DATASET_CSV = 'ijcai_18_train_dataset.csv'
# TEST_DATASET_CSV = 'ijcai_18_test_dataset.csv'

def dataset_prepare(dataset):
    dataset = dataset.drop_duplicates(subset='instance_id')
    dataset = dt.feature_process(dataset)
    dataset = dt.time_hour_encode(dataset)
    dataset = dt.split_shop_feature(dataset)
    dataset = dt.slide_count(dataset)
    dataset = dt.feature_dtype_convert(dataset)
    dataset = dt.item_feature_internal_combine(dataset)
    dataset = dt.user_feature_internal_combine(dataset)
    dataset = dt.user_item_feature_combine(dataset)
    dataset = dt.user_shop_feature_combine(dataset)
    dataset = dt.shop_item_feature_combine(dataset)

    return dataset

def lgbmc_run(dataset):

    train = dataset[(dataset['day'] >= 18) & (dataset['day'] <= 23)]
    validate = dataset[dataset['day'] == 24]
    best_iter = gc.lgbmc(train=train, validate=validate, is_train=True, iteration=20000)

    # Split dataset into train and test.
    train = dataset[dataset.is_trade.notnull()]
    test = dataset[dataset.is_trade.isnull()]
    predicts = gc.lgbmc(train=train, test=test, is_train=False, iteration=best_iter)
    predicts.to_csv('./to/predicts.txt', sep=" ", index=False)

def ffm_run():

    return 0

if __name__ == '__main__':

    """
    # Prepared dataset have been already stored into csv file.
    train = pd.read_csv(os.path.join(DATASET_DIR, TRAIN_DATASET_RAW), sep="\s+")
    train = dataset_prepare(train)
    test = pd.read_csv(os.path.join(DATASET_DIR, TEST_DATASET_RAW), sep="\s+")
    test = dataset_prepare(test)
    # dataset = pd.concat([train, test])
    train = train.drop_duplicates(subset='instance_id')
    train.to_csv('./to/train.txt', sep=" ", index=False)
    test.to_csv('./to/test.txt', sep=" ", index=False)

    # dataset = pd.read_csv(os.path.join(DATASET_DIR, 'dataset.txt'), sep="\s+")
    train = pd.read_csv('./to/train.txt', sep="\s+")
    # Move train['is_trade'] to the last column
    trade_series = train.pop('is_trade')
    train.insert(len(train.columns), 'is_trade', trade_series)
    test = pd.read_csv('./to/test.txt', sep="\s+")
    # train_cols = [c for c in train]
    # test_cols = [c for c in test]
    dataset = pd.concat([train, test])

    lgbmc_run(dataset)
    """
    """
    predicts = pd.read_csv('./to/predicts.txt', sep="\s+")
    predicts['predicted_trade'] = predicts['predicted_score'].apply(
            lambda x: 1 if x >= 0.85 else (0 if x <= 0.15 else x))
    print(predicts[predicts['predicted_trade'] == 0])
    """
    train = pd.read_csv('./to/train.txt', sep="\s+")
    test = train[train['is_trade'] == 1]
    predicts = gc.lgbmc(train=train, test=test, is_train=False, iteration=429)
    predicts.to_csv('./to/validate_predicts.txt', sep=" ", index=False)













