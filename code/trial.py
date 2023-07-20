import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl', 'rb'))

# int_features = [43, 81, 33, 21, 12, 88, 37, 92, 99, 80, 56]
int_features = [50.0, 0, 168, 62.0, 110, 80, 1, 1, 0, 0, 1] #0

cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

df = pd.DataFrame([int_features], columns=cols)
# scaler = StandardScaler()
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
df[numerical_features] = scaler.transform(df[numerical_features])
# categorical_features = ['cholesterol', 'gluc']
# df = pd.get_dummies(df, columns=categorical_features)

print(df.head())

# cols2 = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
# df1= [-0.414535, 0, 0.443452, -0.847873, -0.122182, -0.088238, 0, 0, 1, 0, True, False, False, True, False, False]
# df2 = pd.DataFrame([int_features], columns=cols)

prediction=model.predict_proba(df)
#     # prediction=model.predict(final)
output='{0:.{1}f}'.format(prediction[0][1], 2)

print(prediction)