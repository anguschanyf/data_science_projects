# this code is to visualise and predict used car price based on a dataset obtained from Kaggle. (https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) [license: CC0: Public Domain]


import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max.columns', None)


# import data
data = pd.read_csv("/Users/changus/Downloads/toyota.csv.xls")
print(data.shape)
data.head()

# check the number of missing values
data.isnull().sum()

data.describe()

## EDA
# plot counts grouped by transmission types
sns.countplot(data, x = "transmission")

# show proportion of models
print(data["model"].value_counts() / len(data))
sns.countplot(y = data["model"])

# plot fuel type
sns.countplot(data, x = "fuelType")

plt.figure(figsize=(10,5),facecolor='w') 
sns.barplot(x = "year", y = "price", data = data)
plt.tight_layout()
plt.show()

# scatter plot mileage against price
plt.figure(figsize=(10,5),facecolor='w') 
sns.scatterplot(data=data, x="mileage", y="price", hue="year")
plt.show()

plt.figure(figsize=(15,5),facecolor='w') 
sns.scatterplot(data=data, x="mileage", y="price", hue="fuelType")

sns.pairplot(data)

# compute age and drop year
data["age"] = 2020 - data["year"]
data = data.drop(columns = ["year"])
data.sample(10)


## pre-process for modelling
data_expanded = pd.get_dummies(data)

# standardize all variables
std = StandardScaler()
data_expanded_std = std.fit_transform(data_expanded)
data_expanded_std = pd.DataFrame(data_expanded_std, columns = data_expanded.columns)

# train test split
X_train, X_test, y_train, y_test = train_test_split(data_expanded_std.drop(columns = ['price']), data_expanded_std[['price']])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# modelling
column_names = data_expanded.drop(columns = ['price']).columns

no_of_features = []
r_squared_train = []
r_squared_test = []

for k in range(3, 32, 2):
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train, y_train)
    X_test_transformed = selector.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r_squared_train.append(regressor.score(X_train_transformed, y_train))
    r_squared_test.append(regressor.score(X_test_transformed, y_test))
    
sns.lineplot(x = no_of_features, y = r_squared_train, legend = 'full')
sns.lineplot(x = no_of_features, y = r_squared_test, legend = 'full')

# since the cruve stablizes around 21 variables
selector = SelectKBest(f_regression, k = 21)
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)
column_names[selector.get_support()]

def regression_model(model):
    """
    Will fit the regression model passed and will return the regressor object and the score
    """
    regressor = model
    regressor.fit(X_train_transformed, y_train)
    score = regressor.score(X_test_transformed, y_test)
    return regressor, score

model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])

models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = pd.concat([model_performance, pd.DataFrame({"Features": "Linear", "Model": [model], "Score": [score]})], ignore_index=True)

model_performance

# fitting a linear regression model and checking the model parameters
regressor = sm.OLS(y_train, X_train).fit()
print(regressor.summary())

X_train_dropped = X_train.copy()

# dropping variables for p values higher than 0.05
while True:
    if max(regressor.pvalues) > 0.05:
        drop_variable = regressor.pvalues[regressor.pvalues == max(regressor.pvalues)]
        print("Dropping " + drop_variable.index[0] + " and running regression again because pvalue is: " + str(drop_variable[0]))
        X_train_dropped = X_train_dropped.drop(columns = [drop_variable.index[0]])
        regressor = sm.OLS(y_train, X_train_dropped).fit()
    else:
        print("All p values less than 0.05")
        break
    
# print the regression result
print(regressor.summary())