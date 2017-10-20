import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import r2_score

data = pd.read_csv('../data/train.csv', delimiter=',').drop(['Id'], axis=1)
#test_data = pd.read_csv('../data/test.csv', delimiter=',').drop(['Id'], axis=1)

# Find columns
data_corr = data.corr().abs()
important_columns = [col for col in data_corr.loc[:, data_corr['SalePrice'] >= 0.5].columns]

#Split data
train_data = data.iloc[:int(data.shape[0]*0.8), :][important_columns]
test_data = data.iloc[int(data.shape[0]*0.8):, :][important_columns]

X = train_data.drop(['SalePrice'], axis=1)
Y = train_data['SalePrice']

model = XGBRegressor()
model.fit(X, Y)

prediction = model.predict(test_data.drop(['SalePrice'], axis=1))

print('R2 score: {}'.format(r2_score(test_data['SalePrice'], prediction)))
