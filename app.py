from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('diabetes.csv')
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df['Glucose'].fillna(df['Glucose'].median(), inplace =True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace =True)
df['BMI'].fillna(df['BMI'].median(), inplace =True)

by_Glucose_Age_Insulin_Grp = df.groupby(['Glucose'])
def fill_Insulin(series):
    return series.fillna(series.median())
df['Insulin'] = by_Glucose_Age_Insulin_Grp['Insulin'].transform(fill_Insulin)
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())

by_BMI_Insulin = df.groupby(['BMI'])
def fill_Skinthickness(series):
    return series.fillna(series.mean())
df['SkinThickness'] = by_BMI_Insulin['SkinThickness'].transform(fill_Skinthickness)
df['SkinThickness'].fillna(df['SkinThickness'].mean(),inplace= True)

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print(accuracy_score(y_pred,y_test))

app=Flask(__name__)

@app.route('/')
def gets_connected():
 return(render_template('index.html'))

@app.route('/',methods=['POST'])
def read_data():
 print('Reading data from Form')
 pregnancies=int(request.form['pregnancies'])
 glucose=int(request.form['glucose'])
 bp=int(request.form['bp'])
 st=int(request.form['skinthickness'])
 insulin=int(request.form['insulin'])
 bmi = float(request.form['bmi'])
 dpf=float(request.form['dpf'])
 age=int(request.form['age'])
 name=request.form['name']
 print(pregnancies,glucose,bp,st,insulin,bmi,dpf,age)
 text=classifier.predict(sc.fit_transform([[pregnancies,glucose,bp,st,insulin,bmi,dpf,age]]))
 print(text)
 if text[0]==0:
  k='Negative'
 else:
  k='Positive'
 return(render_template('index.html' ,prediction_output=k))

if __name__=="__main__":
 app.run()