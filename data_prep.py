import pandas as pd
import numpy as np
from functools import reduce
import operator
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', 199)  # or 199

location_science = pd.read_pickle("data/location_science.pkl")
X = location_science[[col for col in location_science.columns.tolist() if col!='price_Close']]
y = location_science['price_Close']
## TODO fitting random forest regression

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

X_array, y_array = X.values, y.values
tscv = TimeSeriesSplit(n_splits=10)
for train_index, test_index in tscv.split(X_array):
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index].reshape(-1, 1), y_array[test_index].reshape(-1, 1)
    X_train_scaler, X_test_scaler = StandardScaler().fit(X_train), StandardScaler().fit(X_test)
    y_train_scaler, y_test_scaler = StandardScaler().fit(y_train), StandardScaler().fit(y_test)
    scaled_X_train, scaled_X_test = X_train_scaler.transform(X_train), X_test_scaler.transform(X_test)
    scaled_y_train, scaled_y_test = y_train_scaler.transform(y_train), y_test_scaler.transform(y_test)
    randForestRegr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators = 100)
    print(len(scaled_X_train), len(scaled_y_train))
    randForestRegr.fit(scaled_X_train, scaled_y_train)
    y_test_pred = randForestRegr.predict(X_test)
    plt.plot(y_test_scaler.inverse_transform(y_test_pred), color='b')
    plt.plot(y_test_scaler.inverse_transform(y_test), y_test, color='r')
    plt.show()
    plt.close()




print(X.shape)

## TODO fitting LASSO to show feature importance
from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# X_scaler, y_scaler = StandardScaler().fit(X), StandardScaler().fit(y.values.reshape(-1, 1))
#scaled_X, scaled_y = X_scaler.transform(X), y_scaler.transform(y.values.reshape(-1, 1))
# regr = linear_model.LinearRegression()
# model = regr.fit(scaled_X, scaled_y)
# model = LassoCV(cv=5, random_state=0).fit(scaled_X, scaled_y)

print("Best alpha using built-in LassoCV: %f" % model.alpha_)
print("Best score using built-in LassoCV: %f" %model.score(scaled_X,scaled_y))
# coef = pd.Series(model.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# imp_coef = coef.sort_values()
# imp_coef = imp_coef[imp_coef!=0]
# print(imp_coef, len(imp_coef))
# imp_coef.plot(kind = "barh")
# fig = plt.gcf()
# fig.set_size_inches(30, 10)
# plt.title("Feature importance using Lasso REIT Close Price Model")
# fig.subplots_adjust(left=0.45)
# plt.tight_layout()
# plt.show()
# fig.savefig('Feature importance using Lasso REIT Close Price Model.png', dpi=100)
# plt.close()






## TODO Compute the correlation matrix
corr = X.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 0, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, cbar_kws={"shrink": .5})
yticks = X.index
xticks = X.index
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.xticks(rotation=90)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title("Empirical Correlation Matrix on Location Sciences Features" )
# plt.savefig("resources/Empirical Correlation Matrix on " + readin_date +'.png' , dpi=f.dpi)
# plt.show()
plt.close()


