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
# ijcai_cvr_loader.py -- Convert given dataset *.txt to *.csv format 
# for visualization and processing .

import os
import sys
import codecs
import numpy as np
import pandas as pd
import sklearn as sk
import csv
import matplotlib.pyplot as plt

# Train slices store path
DATASET_DIR = 'd:/engineering-data/Ali-IJCAI-18-data'

TRAIN_DATASET_RAW = 'round1_ijcai_18_train_20180301.txt'

TEST_DATASET_RAW = 'round1_ijcai_18_test_a_20180301.txt'

TRAIN_DATASET_CSV = 'ijcai_18_train_dataset.csv'
TEST_DATASET_CSV = 'ijcai_18_test_dataset.csv'

# Convert dataset raw txt format into csv.
# args:
#	None
# returns:
#	None
def raw_convert_csv(dataset):
	if dataset == 'training':
		dataset_csv = os.path.join(DATASET_DIR, TRAIN_DATASET_CSV)
		dataset_raw = os.path.join(DATASET_DIR, TRAIN_DATASET_RAW)
	elif dataset == 'test':
		dataset_csv = os.path.join(DATASET_DIR, TEST_DATASET_CSV)
		dataset_raw = os.path.join(DATASET_DIR, TEST_DATASET_RAW)
	else:
		print('no dataset intended.')
		return
	'''
	with open(dataset_csv, 'w', encoding='utf-8', newline='') as outfile:
		csv_writer = csv.writer(outfile, dialect='excel')
		with codecs.open(dataset_raw, 'r', 'utf-8') as infile:
			for line in infile:
				csv_writer.writerow(line)

	# pd.merge(pd.read_csv(os.path.join(DATASET_DIR, TEST_DATASET_RAW),
	sep=' ',usecols=[0,6]),pd.read_csv(os.path.join(DATASET_DIR, TRAIN_DATASET_RAW), 
	sep=' ',usecols=[6,26]).groupby('item_price_level',as_index=False).mean(),
	on='item_price_level',how='left').drop('item_price_level',axis=1)
	.to_csv('baseline.csv',index=False,sep=' ',header=['instance_id','predicted_score'])
	# v = np.loadtxt(dataset_raw, dtype=np.str)
	# print(np.shape(v))
	
	# pd.read_csv(dataset_raw, sep=' ', usecols=[0,6]).to_csv(dataset_csv)
	'''
	'''
	with codecs.open(dataset_raw, 'r', 'utf-8') as infile:
		line = infile.readline()
	columns = line.split(' ')
	with codecs.open("./columns.txt", 'w', 'utf-8') as outfile:
		for i in range(len(columns)):
			column = columns[i]
			outfile.write(str(i) + ':' + column + '\r\n')
	
	# Get is_trade distribution 
	pd.read_csv(dataset_raw, sep=' ', usecols=[26]).to_csv(
			os.path.join(DATASET_DIR, "train_trade.csv"))
	
	# Get item distribution
	cols = [x for x in range(1, 10)]
	cols.append(26)
	pd.read_csv(dataset_raw, sep=' ', usecols=cols).to_csv(
			os.path.join(DATASET_DIR, "train_item.csv"))
	
	# Get user distribution
	cols = [x for x in range(10, 15)]
	cols.append(26)
	pd.read_csv(dataset_raw, sep=' ', usecols=cols).to_csv(
			os.path.join(DATASET_DIR, "train_user.csv"))

	# Get context distribution
	cols = [x for x in range(15, 19)]
	cols.append(26)
	pd.read_csv(dataset_raw, sep=' ', usecols=cols).to_csv(
			os.path.join(DATASET_DIR, "train_context.csv"))
	
	# Get shop distribution
	cols = [x for x in range(19, 27)]
	pd.read_csv(dataset_raw, sep=' ', usecols=cols).to_csv(
			os.path.join(DATASET_DIR, "train_shop.csv"))

	df = pd.read_csv(os.path.join(DATASET_DIR, "train_trade.csv"))
	
	df.loc['trade_sums'] = df.apply(lambda x: x.sum())
	slices = [df.loc['trade_sums']['is_trade'], len(df)]
	activities = ['is_trade=1', 'is_trade=0']
	colors = ['g', 'r']
	plt.pie(slices, labels=activities,
		colors=colors, startangle=90,
		shadow=True, explode=(0.1, 0),
		autopct='%1.1f%%')
	plt.title('is_trade distribution')
	plt.show()
	'''
	df = pd.read_csv(os.path.join(DATASET_DIR, "train_item.csv"))
	item_dict = {}
	
	for item in df['item_id']:
		if item in item_dict:
			item_dict[item] += 1
		else:
			item_dict.update({item: 1})
	
	item_count_dict = {}
	for item in item_dict:
		if item_dict[item] in item_count_dict:
			item_count_dict[item_dict[item]] += 1
		else:
			item_count_dict.update({item_dict[item]: 1})
	
	#print([item_count_dict[c] for c in range(1, 51)])
	'''
	plt.bar([c for c in range(1, 51)], [item_count_dict[c] for c in range(1, 51)],
			label="Item Count", color='g')
	plt.xlabel('item occur times')
	plt.ylabel('occur times counts')
	plt.title('Item Distribution')
	plt.legend()
	plt.show()
	'''
	
	item_list = []
	item_times_list = []
	sorted(item_dict.items(), key=lambda e: e[1], reverse=True)
	for item in item_dict:
		if item_dict[item] > 1300:
			item_list.append(str(int(item))[0:3])
			item_times_list.append(item_dict[item])
	#print(item_list)
	#print(item_times_list)
	plt.bar([id for id in range(len(item_list))], item_times_list, 
			label="Item Count", color='g')
	plt.xlabel('item id')
	plt.ylabel('occur times')
	plt.title('Top20 Items')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	raw_convert_csv('training')


























