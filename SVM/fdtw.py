# dynamic time warping

import pandas as pd
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_colwidth', 200)

import pandas_datareader as pdr
start_date = pd.to_datetime('2000-01-01')
stop_date = pd.to_datetime('2016-03-01')

# close connection to finace.yahoo
#spy = pdr.data.get_data_yahoo('SPY', start_date, stop_date)
#spy.to_csv('spy.csv')
spy = pd.read_csv('spy.csv')
spy_c = spy['Close']

spy['Daily Change'] = pd.Series(spy['Close'] - spy['Open'])
#np.std(spy['Daily Change'])
spy['Overnight Change'] = pd.Series(spy['Open'] - spy['Close'].shift(1))
#np.std(spy['Overnight Change'])

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def dtw_dist(x, y):
    distance, path = fastdtw(x, y, dist = euclidean)
    return distance

tseries = []
tlen = 5
for i in range(tlen, len(spy), tlen):
    pctc = spy['Close'].iloc[(i-tlen):i].pct_change()[1: ].values * 100
    #print(spy['Close'].iloc[(i-tlen):i].pct_change())
    res = spy['Close'].iloc[(i-tlen):(i + 1)].pct_change()[-1 :] * 100
    #print(res)
    tseries.append((pctc, res))
    
dist_pairs = []
for i in range(len(tseries)):
    for j in range(len(tseries)):
        dist = dtw_dist(tseries[i][0], tseries[j][0])
        #print(dist)
        dist_pairs.append((i, j, dist, tseries[i][1], tseries[j][1]))
        
dist_frame = pd.DataFrame(dist_pairs, columns = ['A', 'B', 'Dist', 'A Ret', 'B Ret'])
sf = dist_frame[dist_frame['Dist'] > 0].sort_values(['A', 'B']).reset_index(drop = 1)

sfe = (sf['A'] < sf['B']).any()

# invalid index to scalar variable
#winf = ((sfe['Dist'] <= 1) & (sfe['A Ret'] > 0)).any()
winf = (sfe['Dist'] <= 1).any()
winf = (winf['A Ret'] > 0).any()
print(winf)
