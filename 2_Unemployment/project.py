

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from datetime import datetime
from sklearn.svm import SVR
import calendar
from sklearn.preprocessing import LabelEncoder

"""## Load Dataset"""

read=pd.read_csv("/home/husnain/Desktop/PL/Python/REMOTE_INTERNSHIP/Swift_Solver/2_unemployment/Unemployment in India.csv")
df=pd.DataFrame(read)
df.head()

"""## Data Preprocessing"""

df.columns
#['Region', ' Date', ' Frequency', ' Estimated Unemployment Rate (%)',' Estimated Employed', ' Estimated Labour Participation Rate (%)',  'Area'],

#our target variable :Estimated Unemployment Rate (%)

df.info()

df.describe()

df.isnull().sum()
#no null data already cleaned

df[df.duplicated()]
#no duplicates in this dataset

df['Region'].value_counts()
#Total 28 regions

df['Area'].value_counts()
#only rural and urban areas

df['Frequency'].value_counts()

df['Frequency']="Monthly"
print(df['Frequency'].value_counts())
#Data is misplace by just space

df["Date"]

Date= df["Date"].str.split('-', expand=True)
df["Day"]=Date[0]
df["month"]=Date[1]
df["year"]=Date[2]

"""# Data Visulization

#### Which Region hve hign unemployment
"""

plt.figure(figsize=(15, 10))
sns.histplot(df['Region'], kde=True,color="red")
plt.xticks(rotation=90)
plt.title('Regions')

plt.figure(figsize=(10, 5))
sns.histplot(df['Area'],color="green")
plt.xticks(rotation=90)
plt.title('Area')

plt.bar(df['Date'], df['Estimated Unemployment Rate (%)'])
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate ')
plt.title('Time Series Plot')
plt.xticks(rotation=90)
plt.show()

# Calculate the correlation matrix
correlation_matrix = df.corr()


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap: Correlation Matrix')
plt.show()

"""# Feature Engineering"""

df = pd.get_dummies(df, columns=['Area',"Frequency"])
label_encoder = LabelEncoder()
df['Region'] = label_encoder.fit_transform(df['Region'])

numerical_features = ["Estimated Unemployment Rate (%)","Estimated Employed","Estimated Labour Participation Rate (%)"]

scaler_minmax = MinMaxScaler()
df[numerical_features] = scaler_minmax.fit_transform(df[numerical_features])

df.describe()

"""# Model Training"""

x = df.drop(['Estimated Unemployment Rate (%)',"Frequency_Monthly","Date"], axis=1)
y=df["Estimated Unemployment Rate (%)"]
print(x.columns)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

df.describe()

X_test

"""## Linear Regression Model"""

Linear = LinearRegression()

Linear.fit(X_train, y_train)

y_pred =Linear.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

print(f"R-squared (R²): {r2}")

def linear_fun(list1):
    y_predq=Linear.predict(list1)
    return y_predq

n_features = Linear.coef_.shape[0]
print("Number of features:", n_features)

"""## Decision Tree Regressor"""

DecisionTree = DecisionTreeRegressor()

DecisionTree.fit(X_train, y_train)

y_pred = DecisionTree.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

print(f"R-squared (R²): {r2}")

"""## Random Forest Classifier"""

RandomForest = RandomForestRegressor()

RandomForest.fit(X_train, y_train)

y_pred = RandomForest.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

print(f"R-squared (R²): {r2}")

"""## Save Models"""

#Logestic Regression mode
pickle.dump(Linear , open("logestic.pkl", "wb"))

#Decision Tree classifier
pickle.dump(DecisionTree , open("DecisionTree.pkl", "wb"))

#Random Forest Classifier
pickle.dump(RandomForest, open("RandomForest.pkl", "wb"))



