# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		numpy 1.13.1
#		pandas 0.20.3
#		sklearn 0.19.0
#       lightbgm 2.1.0
# -*- author: Hsingmin Lee
#
# dataset.py -- Transform sample features and fields .
#
import warnings
import lightbgm as lgb
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import csv

warnings.filterwarnings("ignore")

# Train slices store path
DATASET_DIR = 'd:/engineering-data/Ali-IJCAI-18-data'

TRAIN_DATASET_RAW = 'round1_ijcai_18_train_20180301.txt'

TEST_DATASET_RAW = 'round1_ijcai_18_test_a_20180301.txt'

TRAIN_DATASET_CSV = 'ijcai_18_train_dataset.csv'
TEST_DATASET_CSV = 'ijcai_18_test_dataset.csv'

# Convert Unix time stamps into local date. 
def ustamps_to_time(value):
    format = '%Y-%m-%d %H:%M:%S'
    # Get local time.
    value = time.localtime(value)
    # Format local time like 2018-04-26 22:41:29 . 
    date = time.strftime(format, value)
    return date
# Encode user_occupation_id feature.
def user_occupation_encode(x):
    if x == -1 | x == 2003:
        return 1
    elif x == 2002:
        return 2
    else:
        return 3

# Encode user_star_level feature.
def user_starlevel_encode(x):
    if x == -1 | x == 3000:
        return 1
    elif x == 3009 | x == 3010:
        return 2
    else:
        return 3

# Encode shop_score_delivery feature.
def shop_delivery_encode(x):
    if x == -5:
        z = 1
    else:
        z = (x-4.1)*10 + 1

    if (z >= 2) & (z <= 4):
        return 1
    elif (z >= 5) & (z <= 7):
        return 2
    else:
        return 3

# Encode shop_score_service feature.
def shop_service_encode(x):
    if x == -5:
        z = 1
    else:
        z = (x-3.93)*10 + 1

    if (z >= 2) & (z <= 7):
        return 1
    elif (z >= 8) & (z <= 9):
        return 2
    else:
        return 3

# Encode shop_score_description feature.
def shop_description_encode(x):
    if x == -5:
        z = 1
    else:
        z = (x-3.93)*10 + 1

    if (z >= 2) & (z <= 8):
        return 1
    elif (z >= 9) & (z <= 10):
        return 2
    else:
        return 3

# Encode shop_review_positive_rate feature.
def shop_reviewrate_encode(x):
    if x == -1:
        z = 1
    else:
        z = (x-0.714)*50 + 1

    if (z >= 2) & (z <= 12):
        return 1
    elif (z >= 13) & (z <= 15):
        return 2
    else:
        return 3

# Encode hour in time field.
def time_hour_encode(data):
    data['time_hour_segment'] = data['hour'].apply(
            lambda x: 1 if (x<=12) & (x>=7) else(2 if (x<=20) & (x>=13) else 3))
    return data

# Dataset fields and features preprocess.
def feature_process(data):
    le = preprocessing.LabelEncoder()

    # Item field label encode. 
    data['len_item_category'] = data['item_category_list'].map(lambda x: len(str(x).split(';')))
    data['len_item_property'] = data['item_property_list'].map(lambda x: len(str(x).split(';')))

    for i in range(1, 3):
        data['item_category_list' + str(i)] = le.fit_tranform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

    for i in range(10):
        data['item_property_list' + str(i)] = le.fit_transform(data['item_property_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = le.fit_transform(data[col])

    # User field label encode.
    for col in ['user_id']:
        data[col] = le.fit_tranform(data[col])
    data['gender0'] = data['user_gender_id'].apply(lambda x: 1 if x==-1 else 2)
    data['age0'] = data['user_age_level'].apply(lambda x: 1 if x==1000|x==1001|x==-1 else 2)
    data['occupation0'] = data['user_occupation_id'].apply(user_occupation_encode)
    data['star0'] = data['user_star_level'].apply(user_starlevel_encode)

    # Context field encode.
    data['realtime'] = data['context_timestamp'].apply(ustamps_to_time)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['day'] = data['realtime'].dt.day
    data['hour'] = data['realtime'].dt.hour

    data['len_predict_category_property'] = data['predict_category_property'].map(
            lambda x: len(str(x).split(';')))
    for i in range(5):
        data['predict_category_property' + str(i)] = le.fit_transform(data['predict_category_property'].map(
                lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

    data['context_page0'] = data['context_page_id'].apply(
            lambda x: 1 if x==4001 | x==4002 | x==4003 | x==4004 | x==4007 else 2)

    # Shop filed encode.
    for col in ['shop_id']:
        data[col] = le.fit_transform(data[col])
    data['shop_score_delivery0'] = data['shop_score_delivery'].apply(
            lambda x: 0 if 0.98 >= x >= 0.96 else 1)

    return data

# Split shop fields and features into segments.
def split_shop_feature(data):
    data['shop_score_delivery'] = data['shop_score_delivery']*5
    data = data[data['shop_score_delivery'] != -5]
    data['shop_delivery_segment'] = data['shop_score_delivery'].apply(shop_delivery_encode)
    delivery_segment_counts = data.shop_delivery_segment.value_counts()

    data['shop_score_service'] = data['shop_score_service']*5
    data = data[data['shop_score_service'] != -5]
    data['shop_service_segment'] = data['shop_score_service'].apply(shop_service_encode)
    service_segment_counts = data.shop_service_segment.value_counts()

    data['shop_score_description'] = data['shop_score_description']*5
    data = data[data['shop_score_description'] != -5]
    data['shop_description_segment'] = data['shop_score_description'].apply(shop_description_encode)
    description_segment_counts = data.shop_description_segment.value_counts()

    data = data[data['shop_review_positive_rate'] != -1]
    data['shop_reviewrate_segment'] = data['shop_review_positive_rate'].apply(shop_reviewrate_encode)
    reviewrate_segment_counts = data.shop_reviewrate_segment.value_counts()

    # Get shop feature data normal interval.
    data['shop_normal'] = data.apply(lambda x: 1 if (x.shop_delivery_segment == 3) & (x.shop_description_segment == 3)
            & (x.shop_reviewrate_segment == 3) & (x.shop_service_segment == 3) else 0, axis=1)
    del data['shop_delivery_segment']
    del data['shop_description_segment']
    del data['shop_reviewrate_segment']
    del data['shop_service_segment']
    return data

# Get user_id, item_id, shop_id counts in a slected week.
def slide_count(data):
    for d in range(19, 26):
        df1 = data[data['day'] == d-1]
        df2 = data[data['day'] == d]
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['item_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['shop_id'].to_dict()
        df2['user_cnt1'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cnt1'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cnt1'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))

        df2 = df2[['user_cnt1', 'item_cnt1', 'shop_cnt1', 'instance_id']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')

    for d in range(19, 26):
        df1 = data[data['day'] < d]
        df2 = data[data['day'] == d]
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['item_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['shop_id'].to_dict()
        df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cntx'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cntx'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))

        df2 = df2[['user_cntx', 'item_cntx', 'shop_cntx', 'instance_id']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')

    return data

# Features combine.
def feature_combine(data):
    for col in ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
        data[col] = data[col].apply(lambda x: 0 if x==-1 else 1)

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                'shop_review_num_level', 'shop_star_level']:
        data[col] = data[col].astype(str)

    # Item feature combination.
    data['sales_price'] = data['item_sales_level'] + data['item_price_level']
    data['sales_collected'] = data['item_sales_level'] + data['item_collected_level']
    data['price_collected'] = data['item_price_level'] + data['item_collected_level']

    # User feature combination.
    data['gender_age'] = data['user_gender_id'] + data['user_age_level']
    data['gender_occupation'] = data['user_gender_id'] + data['user_occupation_id']
    data['gender_star'] = data['user_gender_id'] + data['user_star_level']
    # data['age_occupation'] = data['user_age_level'] + data['user_occupation_id']
    # data['age_star'] = data['user_age_level'] + data['user_star_level']
    # data['occupation_star'] = data['user_occupation_id'] + data['user_star_level']

    # Shop feature combination.
    data['review_star'] = data['shop_review_num_level'] + data['shop_star_level']

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'sales_price', 'sales_collected', 'price_collected',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                'gender_age', 'gender_occupation', 'gender_star',
                'shop_review_num_level', 'review_star']:
        data[col] = data[col].astype(int)

    del data['review_star']

    return data

# Item feature process which is the key for model.
def item_feature_process(data):
    # Process item features in detail.
    # item_id combine with other features.
    item_count = data.groupby(by=['item_id'], as_index=False)['instance_id'].agg({'item_count': 'count'})
    data = pd.merge(data, item_count, on=['item_id'], how='left')
    for col in ['item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level']:
        col_count = data.groupby(by=[col, 'item_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_item_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'item_id'], how='left')
        data[str(col)+'_item_prob'] = data[str(col)+'_item_count']/data['item_count']

    del data['item_count']

    # item_brand_id combine with other features.
    item_brand_count = data.groupby(by=['item_brand_id'], as_index=False)['instance_id'].agg(
            {'item_brand_count': 'count'})
    data = pd.merge(data, item_brand_count, on=['item_brand_id'], how='left')
    for col in ['item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        col_count = data.groupby(by=[col, 'item_brand_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_brand_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'item_brand_id'], how='left')
        data[str(col) + '_brand_prob'] = data[str(col) + '_brand_count']/data['item_brand_count']
    del data['item_brand_count']

    # item_city_id combine with other features.
    item_city_count = data.groupby(by=['item_city_id'], as_index=False)['instance_id'].agg(
            {'item_city_count': 'count'})
    data = pd.merge(data, item_city_count, on=['item_city_id'], how='left')
    for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        col_count = data.groupby(by=[col, 'item_city_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_city_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_city_prob'] = data[str(col) + '_city_count']/data['item_city_count']
    del data['item_city_count']

    # item_price_level combine with other features.
    item_price_count = data.groupby(by=['item_price_level'], as_index=False)['instance_id'].agg(
            {'item_price_count': 'count'})
    data = pd.merge(data, item_price_count, on=['item_price_level'], how='left')
    for col in ['item_sales_level', 'item_collected_level', 'item_pv_level']:
        col_count = data.groupby(by=[col, 'item_price_level'], as_index=False)['instance_id'].agg(
                {str(col) + '_price_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'item_price_level'], how='left')
        data[str(col) + '_price_prob'] = data[str(col) + '_price_count']/data['item_price_count']
    del data['item_price_count']

    # item_sales_level combine with other features.
    item_sales_count = data.groupby(by=['item_sales_level'], as_index=False)['instance_id'].agg(
            {'item_sales_count': 'count'})
    data = pd.merge(data, item_sales_count, on=['item_sales_level'], how='left')
    for col in ['item_collected_level', 'item_pv_level']:
        col_count = data.groupby(by=[col, 'item_sales_level'], as_index=False)['instance_id'].agg(
                {str(col) + '_sales_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'item_sales_level'], how='left')
        data[str(col) + '_sales_prob'] = data[str(col) + '_sales_count']/data['item_sales_count']
    del data['item_sales_count']

    # item_collected_level combine with other features.
    item_collected_count = data.groupby(by=['item_collected_level'], as_index=False)['instance_id'].agg(
            {'item_collected_count': 'count'})
    data = pd.merge(data, item_collected_count, on=['item_collected_level'], how='left')
    for col in ['item_pv_level']:
        col_count = data.groupby(by=[col, 'item_collected_level'], as_index=False)['instance_id'].agg(
                {str(col) + '_collected_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'item_collected_level'], how='left')
        data[str(col) + '_collected_prob'] = data[str(col) + '_collected_count']/data['item_collected_count']
    del data['item_collected_count']

    return data

# Process user features in detail.
def user_feature_process(data):
    # Process user features in detail.
    # user_id combine with other features.
    user_count = data.groupby(by=['user_id'], as_index=False)['instance_id'].agg({'user_count': 'count'})
    data = pd.merge(data, user_count, on=['user_id'], how='left')
    for col in ['user_gender_id', 'user_occupation_id', 'user_age_level', 'user_star_level']:
        col_count = data.groupby(by=[col, 'user_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_user_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'user_id'], how='left')
        data[str(col)+'_user_prob'] = data[str(col)+'_user_count']/data['user_count']

    del data['user_count']

    # user_gender_id combine with other features.
    user_gender_count = data.groupby(by=['user_gender_id'], as_index=False)['instance_id'].agg(
            {'user_gender_count': 'count'})
    data = pd.merge(data, user_gender_count, on=['user_gender_id'], how='left')
    for col in ['user_occupation_id', 'user_age_level', 'user_star_level']:
        col_count = data.groupby(by=[col, 'user_gender_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_gender_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'user_gender_id'], how='left')
        data[str(col)+'_gender_prob'] = data[str(col)+'_gender_count']/data['user_gender_count']

    del data['user_count']

    # user_occupation_id combine with other features.
    user_occupation_count = data.groupby(by=['user_occupation_id'], as_index=False)['instance_id'].agg(
            {'user_occupation_count': 'count'})
    data = pd.merge(data, user_occupation_count, on=['user_occupation_id'], how='left')
    for col in ['user_age_level', 'user_star_level']:
        col_count = data.groupby(by=[col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_occupation_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'user_occupation_id'], how='left')
        data[str(col)+'_occupation_prob'] = data[str(col)+'_occupation_count']/data['user_occupation_count']

    del data['user_occupation_count']

    # user_age_level combine with other features.
    user_age_count = data.groupby(by=['user_age_level'], as_index=False)['instance_id'].agg(
            {'user_occupation_count': 'count'})
    data = pd.merge(data, user_occupation_count, on=['user_occupation_id'], how='left')
    for col in ['user_age_level', 'user_star_level']:
        col_count = data.groupby(by=[col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
                {str(col) + '_occupation_count': 'count'})
        data = pd.merge(data, col_count, on=[col, 'user_occupation_id'], how='left')
        data[str(col)+'_occupation_prob'] = data[str(col)+'_occupation_count']/data['user_occupation_count']

    del data['user_occupation_count']





