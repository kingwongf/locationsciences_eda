import pandas as pd
import numpy as np
from functools import reduce
import operator
import matplotlib.pyplot as plt
import seaborn as sns
# pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', -1)  # or 199

location_science = pd.read_pickle("data/location_science.pkl")
#(location_science[['price_Close']])
X = location_science.copy()
location_science['forward_return_side'] = np.sign(np.log(location_science.price_Close).diff().shift(-1))
# print(X_price[['price_Close', 'forward_return']])

X = location_science[[col for col in location_science.columns.tolist() if col!='forward_return_side']].sort_index()[location_science.index <'2019-03-31']
y = location_science['forward_return_side'].dropna()

## TODO fitting random forest regression

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.linear_model import LassoCV

from sklearn import datasets, linear_model


X_array, y_array = X.values, y.values
tscv = TimeSeriesSplit(n_splits=5)



def best_RandForestReg(X, y):
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0,n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=False, verbose=False).fit(X, y)
    return rfr

def best_RandForestClas(X, y):
    gsc_3 = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, verbose=0,n_jobs=-1)

    ## TODO remove temp fix after resolve GridSearch  ValueError: Unknown label type: 'continuous'
    '''
    scores_3 = [[(RandomForestClassifier(max_depth, n_estimators).fit(X, y).predict(X) - y)**0.5, n_estimators, max_depth]
                for n_estimators in [10, 50, 100, 1000]
                for max_depth in range(3,7)]
    scores_3 = pd.DataFrame(scores_3, columns=['score', 'max_depth', 'n_estimators']).sort_values(by="score", ascending=True)
    best_params_3 = scores_3[0][]
    '''
    grid_result_3 = gsc_3.fit(X, y)
    best_params_3 = grid_result_3.best_params_
    rfc = RandomForestClassifier(max_depth=best_params_3["max_depth"], n_estimators=best_params_3["n_estimators"], random_state=False, verbose=False).fit(X, y)
    return rfc

def best_AdaBoostClas(X, y):
    gsc_4 = GridSearchCV(
        estimator=AdaBoostClassifier(),
        param_grid={
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0,n_jobs=-1)

    grid_result_4 = gsc_4.fit(X, y)
    best_params_4 = grid_result_4.best_params_
    adbc = AdaBoostClassifier(n_estimators=best_params_4["n_estimators"], learning_rate=0.1, random_state=False).fit(X, y)
    return adbc

scaled_X = StandardScaler().fit(X_array).transform(X_array)
model = best_RandForestClas(scaled_X,y_array.reshape(-1, 1))
pred_y = model.predict(scaled_X)
mse = mean_squared_error(y_array, pred_y)
print(mse)
exit()






## TODO: 1 day forward return side prediction
for train_index, test_index in tscv.split(X_array):
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index].reshape(-1, 1), y_array[test_index].reshape(-1, 1)
    X_train_scaler, X_test_scaler = StandardScaler().fit(X_train), StandardScaler().fit(X_test)
    # y_train_scaler, y_test_scaler = StandardScaler().fit(y_train), StandardScaler().fit(y_test)
    scaled_X_train, scaled_X_test = X_train_scaler.transform(X_train), X_test_scaler.transform(X_test)
    # scaled_y_train, scaled_y_test = y_train_scaler.transform(y_train), y_test_scaler.transform(y_test)
    scaled_y_train, scaled_y_test = y_train, y_test

    ''' LASSO is completely flat, useless'''
    # model = LassoCV(cv=5, random_state=0).fit(scaled_X_train, scaled_y_train)
    # predict_y_test = model.predict(scaled_X_test)


    ## Random Forest Regressor
    model2 = best_RandForestReg(scaled_X_train, scaled_y_train)
    predict_y_test_2 = model2.predict(scaled_X_test)
    mse_2 = mean_squared_error(scaled_y_test, np.sign(predict_y_test_2))
    # f1_2 = f1_score(scaled_y_test, np.sign(predict_y_test_2))

    ## Random Forest Classifier
    model3 = best_RandForestClas(scaled_X_train, scaled_y_train)
    predict_y_test_3_class = model3.predict(scaled_X_test)
    mse_3 = mean_squared_error(scaled_y_test, predict_y_test_3_class)
    # f1_3 = f1_score(scaled_y_test, predict_y_test_3_class)

    ## AdaBoost Classifier
    model4 = best_AdaBoostClas(scaled_X_train, scaled_y_train)
    predict_y_test_4_class= model4.predict(scaled_X_test)
    mse_4 = mean_squared_error(scaled_y_test, predict_y_test_4_class)
    # f1_4 = f1_score(scaled_y_test, predict_y_test_4_class)


    '''
    plt.plot(y_test_scaler.inverse_transform(predict_y_test_2), color='g', label='Random Forest Regressor MSE: %s' %mse_2.round(), linewidth="0.5")
    plt.plot(y_test_scaler.inverse_transform(predict_y_test_3_class), color='b', label='Random Forest Classifer MSE: %s' %mse_3.round(), linewidth="0.5")
    plt.plot(y_test_scaler.inverse_transform(predict_y_test_4_class), color='r', label='AdaBoost Classifer MSE: %s' %mse_4.round(), linewidth="0.5")
    '''
    plt.plot(predict_y_test_2, color='g', label='Random Forest Regressor MSE: %s' %mse_2.round(), linewidth="0.5")
    plt.plot(predict_y_test_3_class, color='b', label='Random Forest Classifer MSE: %s' %mse_3.round(), linewidth="0.5")
    plt.plot(predict_y_test_4_class, color='r', label='AdaBoost Classifer MSE: %s' %mse_4.round(), linewidth="0.5")
    plt.plot(scaled_y_test, color='black', linewidth="0.5")
    plt.title("Forest 1 day Forward Return Side model")
    plt.legend()
    plt.savefig("Forest 1 day Forward Return Side model.png")
    plt.close()


'''
# print(X.shape)

## TODO fitting LASSO to show feature importance
from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

X_scaler, y_scaler = StandardScaler().fit(X), StandardScaler().fit(y.values.reshape(-1, 1))
scaled_X, scaled_y = X_scaler.transform(X), y_scaler.transform(y.values.reshape(-1, 1))
regr = linear_model.LinearRegression()
model = regr.fit(scaled_X, scaled_y)
model = LassoCV(cv=5, random_state=0).fit(scaled_X, scaled_y)

#print("Best alpha using built-in LassoCV: %f" % model.alpha_)
#print("Best score using built-in LassoCV: %f" %model.score(scaled_X,scaled_y))
coef = pd.Series(model.coef_, index = X.columns)
#print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


imp_coef = coef.sort_values()
imp_coef = imp_coef[imp_coef!=0]
# print(imp_coef, len(imp_coef))
imp_coef.plot(kind = "barh")
fig = plt.gcf()
fig.set_size_inches(30, 10)
plt.title("Feature importance using Lasso REIT Close Price Model")
fig.subplots_adjust(left=0.45)
# plt.show()
fig.savefig('Feature importance using Lasso REIT Close Price Model.png', dpi=100)
plt.close()


## TODO Compute the correlation matrix
corr = X.corr()
corr = corr
absCorr = corr.abs()

# extract upper triangle without diagonal with k=1
sol = (absCorr.where(np.triu(np.ones(absCorr.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False)).to_frame()
sol['pairs'] = sol.index
sol = sol.set_index(np.arange(len(sol.index)))

# print(sol)
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
# plt.savefig("resources/Empirical Correlation Matrix on " +'.png' , dpi=f.dpi)
#plt.show()
plt.close()


## TODO PCA Decomposition

from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(X)


# print(pca.explained_variance_ratio_)

# print(pca.singular_values_)


## TODO plotting top features with REIT price

top_x_feat = X[['demographics_%visitorsChinatownPost-war estates, limited means', 'demographics_%visitorsFitzroviaSocialising young renters']]

top_x_feat = top_x_feat*1000
top_x_feat['demographics_%visitorsFitzroviaSocialising young renters'] = top_x_feat['demographics_%visitorsFitzroviaSocialising young renters']/6
top_x_feat['demographics_%visitorsChinatownPost-war estates, limited means'] = top_x_feat['demographics_%visitorsChinatownPost-war estates, limited means']*10
top_x_feat.plot()
y.plot()
fig = plt.gcf()
plt.title('Scaled Top Features with REIT Price')
fig.savefig('Scaled Top Features with REIT Price.png', dpi=100)
plt.show()
plt.close()


## TODO retain only daily data
daily_X = X[[col for col in X.columns.tolist() if "visit" in col and "demographics_" not in col]]


daily_X_scaler, y_scaler = StandardScaler().fit(daily_X), StandardScaler().fit(y.values.reshape(-1, 1))
daily_scaled_X, scaled_y = daily_X_scaler.transform(daily_X), y_scaler.transform(y.values.reshape(-1, 1))
regr = linear_model.LinearRegression()
model = regr.fit(daily_scaled_X, scaled_y)
model = LassoCV(cv=5, random_state=0).fit(daily_scaled_X, scaled_y)

print("Best alpha using built-in LassoCV: %f" % model.alpha_)
print("Best score using built-in LassoCV: %f" %model.score(daily_scaled_X,scaled_y))
coef = pd.Series(model.coef_, index = daily_X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


imp_coef = coef.sort_values()
imp_coef = imp_coef[imp_coef!=0]
print(imp_coef, len(imp_coef))
imp_coef.plot(kind = "barh")
fig = plt.gcf()
fig.set_size_inches(30, 10)
plt.title("Feature importance using Lasso with Daily Freq Features REIT Close Price Model")
fig.subplots_adjust(left=0.45)
# plt.show()
fig.savefig('Feature importance using Lasso with Daily Freq Features REIT Close Price Model.png', dpi=100)
plt.close()


'''
