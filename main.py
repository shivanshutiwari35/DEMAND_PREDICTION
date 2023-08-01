import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('C:/Users/shivanshu tiwari/PycharmProjects/DEMAND_PREDICTION/PRODUCT.csv')

print(data.head())

data = data.dropna()
#print(data)

#fig = px.scatter(data, px)
print(data.corr())

correlations = data.corr(method = 'pearson')
plt.figure(figsize=(15,12))
sns.heatmap(correlations,cmap = "coolwarm", annot = True)
plt.show()

x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]
data = x
target = y
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size= .50, random_state=0)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=44)

model.fit(data_train, target_train)
predictions = model.predict(data_test)

print(predictions)
print(data_train)




