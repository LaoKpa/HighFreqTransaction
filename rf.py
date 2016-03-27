from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import math

import os
os.chdir("~/Desktop/finance")
train1 = pd.read_csv("train_set_midprice.csv")

labels = train1['Y_midprice'].values
print(labels)
featurelist = range(0,train1.shape[1]-2)
features = train1.ix[:,featurelist].values
print(features)
ntree_list = [50, 100, 250, 500]#[100, 250, 500, 1000]#[100, 250, 500, 1000]
depth = [10, 25, 50]
p = len(train1.columns) - 2 #127

n_features = [5, math.floor(p**(.5)), p//20, p//10, p//5, p//2]
print(n_features)
param_grid = dict(n_estimators = ntree_list, max_features = n_features, max_depth = depth)

cv = StratifiedKFold(labels, n_folds = 3, random_state = 20151204)
grid = GridSearchCV(RandomForestClassifier(n_jobs = 3), param_grid=param_grid, cv=cv, verbose = 5, n_jobs = 3)
grid.fit(features, labels)
print(grid.grid_scores_)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


#write grid
with open('~/Desktop/finance/rf_grid.pkl', "wb") as fp:
    pickle.dump(grid, fp)

