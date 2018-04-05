
# recommendationTest.py

import recommendations

itemsim = recommendations.calculateSimilarItems(recommendations.critics)

'''
recommendedList = recommendations.getRecommendedItems(recommendations.critics, itemsim, 'Toby')

print(recommendedList)
'''

prefs = recommendations.loadMovieLens()
print(prefs['87'])





















