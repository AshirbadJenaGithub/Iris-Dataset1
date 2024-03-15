from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
iris=load_iris()
data=pd.DataFrame(iris.data,columns=iris.feature_names)
data.rename(columns={'sepal length (cm)': 'sepal'}, inplace=True)
data.rename(columns={'sepal width (cm)': 'sepalwidth'}, inplace=True)
x=data.drop('sepalwidth',axis=1)
y=data['sepalwidth']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
knn= KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
plt.scatter(prediction,y_test,cmap='virdis')
