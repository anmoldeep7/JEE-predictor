import pandas as pd
import pickle

dataset = pd.read_csv('Percentile.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

#Saving the model to disk
pickle.dump(regressor, open('model.pkl','wb'))

#Loading model to compare results
model = pickle.load(open('model.pkl','rb'))
