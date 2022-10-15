import flask
from  flask import request,jsonify
import mysql.connector
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from requests import Response
from sqlalchemy import create_engine
import pymysql
import utilities as util

app = flask.Flask(__name__)

app.config["DEBUG"] = True
CORS(app)
#Read the churn data
df = pd.read_csv('/home/ubuntu/Desktop/Python/total_churn_data.csv')
columns = df.columns.values.tolist()
#Drop the null values
df=df.dropna()

#Convert the string columns into numeric
df = pd.get_dummies(df, prefix='', prefix_sep='',columns=['sourcing_channel','residence_area_type'])
X = df.drop(['renewal'], axis=1)
#Actual result for training
y = df['renewal']

#Split the training and test data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  

#Split the data in 80:20 ratio for training and testing the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = RandomForestClassifier(n_estimators=100) 

#Drop the id from the trainign data
X_train = X_train.drop(['id'], axis=1)
#Store the test dataset's id column in another dataframe
df_id = X_test['id']
#Drop the id from test dataset
X_test = X_test.drop(['id'], axis=1)

classifier.fit(X_train, y_train) 
predictions = classifier.predict(X_test)
pd_pred = pd.DataFrame(data=predictions, columns=['predictions'], 
                            index=X_test.index.copy())
#Merge the predictions with the test data                            
df_out = pd.merge(X_test, pd_pred, how ='left', left_index=True, 
                 right_index=True)
#Merge the id bac to the dataframe with predictions                 
df_out1 = pd.merge(df_id, df_out, how ='left', left_index=True, 
                 right_index=True)
#Create a dataframe of id and predictions only                 
df_id_pred=pd.DataFrame(df_out1[['id','predictions']])
df_id_pred.to_csv('/home/ubuntu/Desktop/Python/pred.csv',sep=',',index=False)


# Connect to the mysql database
connection = pymysql.connect(host='localhost',
                         user='root',
                         password='Root123$',
                         db='prediction')

# create cursor
cursor=connection.cursor()
print(connection)
df_id_pred.insert(0, 'TimeStamp', pd.to_datetime('now').replace(microsecond=0))

print(df_id_pred.head())
cols = "`,`".join([str(i) for i in df_id_pred.columns.tolist()])
print(cols)
# Insert DataFrame records one by one into policy_prediction table
import datetime
print(datetime.datetime.now())
current_timestamp=datetime.datetime.now()
for i,row in df_id_pred.iterrows():
    sql = "INSERT INTO `policy_prediction` (`" +cols +"`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
    cursor.execute(sql, tuple(row))
    connection.commit()

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, predictions) 
df_cm = pd.DataFrame({'False':[],'True':[]})
array=[]
for rec in cm:
    print (rec)
    false_value=rec[0]
    true_value=rec[1]
    array.append([false_value,true_value])

df_cm=pd.DataFrame(array, columns = ['False','True'])
def predict_for_policy(id ):
    rslt = util.getPredictionData(id)

    dfpred = pd.DataFrame(columns=X_test.columns)
    # Add the dataset to be predicted
    df_rslt = pd.DataFrame(rslt,index=[0])
    print(df_rslt)
    df_rslt = pd.get_dummies(df_rslt, prefix='', prefix_sep='',columns=['sourcing_channel','residence_area_type'])

    dfpred = dfpred.append(df_rslt, ignore_index=True)
    dfpred = dfpred.fillna(0)
    dfpred=dfpred.drop("id",axis=1)
    print(dfpred)
    prediction = classifier.predict(dfpred)
    print(prediction)
    
    return prediction[0]


@app.route('/testchurn',methods=['GET'])   
def test1_churn():
    return (df_cm.to_json())    


@app.route('/churn',methods=['GET'])   
def test1_id():

    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."
    prediction =predict_for_policy(id)
    result = [
        {'id': id,
     'prediction':prediction
     }
    ]
    return jsonify(result)

app.run()