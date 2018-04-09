
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(4,5), columns=['A', 'B', 'C', 'D', 'E'])


df['ROW_SUM'] = df.apply(lambda x: x.sum(), axis=1)
df.loc['COL_SUM'] = df.apply(lambda x: x.sum())

print(df)





























