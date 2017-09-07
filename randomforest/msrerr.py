
import pandas as pd
import numpy as np

a = pd.Series([10, 10, 10, 10])
b = pd.Series([12, 8, 8, 12])

print('Mean Square Root: ')
print(np.sqrt(np.mean((b - a) **2))/ np.mean(a))

print('Mean Delta :')
print((b - a).mean())


train_index = []
test_index = []

for i in range(100):
	result = np.random.choice(2, p = [.65, .35])
	if result == 1:
		test_index.append(i)
	else:
		train_index.append(i)

#train_index = [1, 2, 3, 4]
#train_list = [0, 1, 2, 3, 4, 5, 6, 7]
print(train_index)
print(test_index)

df1 = pd.DataFrame({'name': ['Alfred', 'Lucy', 'Cathy', 'John', 'Mark'], 'data1': range(5)})

df2 = pd.DataFrame({'name': ['Mike', 'Samuel', 'Steff', 'Jeff', 'Frank'], 'data2': range(5)})

print(df1)
print(df2)
print(pd.merge(df1, df2, on = 'name'))

print(pd.merge(df1, df2, left_index = True, right_index = True))

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y)

SVC(C = 1.0, cache_size = 200, class_weight = None, coef0 = 0.0, decision_function_shape = None, degree = 3, gamma = 'auto', kernel = 'rbf', max_iter = -1, probability = False, random_state = None, shrinking = True, tol = 0.001, verbose = False)

print (clf.predict([[-0.8, -1]]))
