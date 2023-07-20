import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
import pickle

df = pd.read_csv("C:/Users/KIIT/Desktop/Study/KiiT/Semester-VI/Minor-Project/code/cardio_train.csv", sep=';')


# Drop the id column
df = df.drop('id', axis=1) #change

# Convert age to years
df['age'] = df['age'] // 365.25

# Convert gender to binary
df['gender'] = df['gender'] % 2


# Scale the numerical features
scaler = StandardScaler()
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
scaler.fit(df[numerical_features])
df[numerical_features] = scaler.transform(df[numerical_features])
pickle.dump(scaler,open('scaler.pkl','wb'))


X = df.drop('cardio', axis=1)
y = df['cardio']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print(df[:1])

y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy of the model is: {accuracy:.2%}')


# EXPORTING MODEL
pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
