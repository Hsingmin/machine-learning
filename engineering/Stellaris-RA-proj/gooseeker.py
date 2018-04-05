
# %load D:/python_work/machine-learning/gooseeker.py

# GooSeeker.py

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
%matplotlib inline

pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.precision", 3)

# Use the file location of GooSeeker.csv
CSV_PATH = r"d:/python_work/machine-learning/gooseeker_utf8.txt"

df = pd.read_csv(CSV_PATH)
df.columns

# df.head().T

su = df

len(su)

su['房型']

# 获取房型-面积-楼层-年代-租金 信息
def parse_info(row):
    # 处理房型信息
    if '室' in row:
        br, lr= row.split('室')[:2]
        lr, nr = lr.split('厅')[:2]
        return pd.Series({'Beds': br, 'Living': lr})
    elif '平米' in row:
        sqr, nr = row.split('平米')[:2]
        return pd.Series({'Square': sqr})
    elif '楼层' in row:
        lyr, nr = row.split('楼层')[:2]
        return pd.Series({'Floor': lyr})
    elif '元' in row:
        rt, nr = row.split('元')[:2]
        return pd.Series({'Rents': rt})


#attr = ((((su['房型'].apply(parse_info))\
#         .join(su['面积'].apply(parse_info)))\
#         .join(su['楼层'].apply(parse_info)))\
#         .join(su['地铁'].apply(lambda x : 1 if '距离' in x else 0))).join(su['租金'].apply(parse_info))

# 获取地铁信息，缺失值NaN填充0

rail = su['地铁'].fillna('0')
rail.name = 'Rail'

sudf = ((((su['房型'].apply(parse_info))\
         .join(su['面积'].apply(parse_info)))\
         .join(su['楼层'].apply(parse_info)))\
         .join(rail))\
         .join(su['租金'].apply(parse_info))
    
sudf.loc[:, 'Floor'] = sudf['Floor'].map(lambda x : 0 if '低' in x else (1 if '中' in x else 2))

sudf.loc[:, 'Rail'] = sudf['Rail'].map(lambda x : 1 if '距离' in x else 0)

# data type reverse
sudf.loc[:, 'Beds'] = sudf['Beds'].astype(int)
sudf.loc[:, 'Living'] = sudf['Living'].astype(int)
sudf.loc[:, 'Square'] = sudf['Square'].astype(int)
sudf.loc[:, 'Rents'] = sudf['Rents'].astype(int)
sudf.loc[:, 'Floor'] = sudf['Floor'].astype(int)
sudf.loc[:, 'Rail'] = sudf['Rail'].astype(int)

# Relationship of Rents-to-Beds and Rail
import patsy
import statsmodels.api as sm

f = 'Rents ~ Beds + Rail'
y, X = patsy.dmatrices(f, sudf, return_type = 'dataframe')

results = sm.OLS(y, X).fit()
print(results.summary())

X.head()

to_pred_idx = X.iloc[0].index
to_pred_zeros = np.zeros(len(to_pred_idx))
tpdf = pd.DataFrame(to_pred_zeros, index = to_pred_idx, columns = ['value'])

# predict the Rents
tpdf.loc['Intercept'] = 1
tpdf.loc['Beds'] = 2
tpdf.loc['Rail'] = 0

results.predict(tpdf['value'])





























