import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from transformer_c import CombinedAttributesAdder
from transformer_c import DataFrameSelector
from transformer_c import CustomBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion






DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url = HOUSING_URL,housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path , "housing.csv")
    return pd.read_csv(csv_path)



def split_train_test(data , test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices] , data.iloc[test_indices]


def display_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())


#fetch_housing_data()

housing = load_housing_data()
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

#housing.hist(bins=50, figsize=(20, 15))
#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.01)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
         s=housing["population"]/100, label="population",
         c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
     )


corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
#plt.show()

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)


imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
print(imputer.statistics_)




num_pipeline = Pipeline(
    [
        ('imptuter',Imputer(strategy="median")),
        ('attribs_adder',CombinedAttributesAdder()),
        ('std_scaler',StandardScaler()),
    ]
)

housing_num_tr = num_pipeline.fit_transform(housing_num)


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline(
    [
        ('selector',DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler',StandardScaler()),
    ]
)

cat_pipeline = Pipeline(
    [
        ('selector',DataFrameSelector(cat_attribs)),
        ('label_binarizer',CustomBinarizer()),
    ]
)

full_pipeline = FeatureUnion(
    transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ]
)

housing_prepared = full_pipeline.fit_transform(housing)

print('begin...........')

print(housing_prepared)

print('end.............')


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

print('already done!!')

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels ,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

housing_predictions1 = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_predictions1)
tree_rmse = np.sqrt(tree_mse)

print("tree rmse begin...")
print(tree_rmse)
print('tree rmse end...')


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared,housing_labels,scoring="neg_mean_squared_error", cv=10)
cross_rmse_scores = np.sqrt(-scores)
print(cross_rmse_scores)
display_scores(cross_rmse_scores)

lin_scores = cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(lin_rmse_scores)
display_scores(lin_rmse_scores)


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)

random_scores = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error", cv=10)
random_rmse_scores = np.sqrt(-random_scores)
print(random_rmse_scores)
display_scores(random_rmse_scores)



'''
#print(housing_num.median().values())
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns)

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(encoder.classes_)

onehot_encoder = OneHotEncoder()
housing_cat_1hot = onehot_encoder.fit_transform(housing_cat.values.reshape(-1,1))
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())

bin_encoder = LabelBinarizer()
housing_cat_1hot_1step = bin_encoder.fit_transform(housing_cat)
print("==thisistheend==")
print(housing_cat_1hot_1step)


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
'''

