from flask import Flask
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def predict():
    
    try:
        
        json_ = request.json
        user_id = json_[0]['account_id']
        custom_model = joblib.load('custom_models/' + str(user_id) + '_model.pkl') 
        columns = joblib.load('columns.pkl')

        df = pd.DataFrame(json_)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['weekday'] = df['date'].dt.dayofweek
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df = df.reindex(columns=columns, fill_value=0)
        print(df)
        if custom_model:
            print('using custom model for user ' + str(user_id))
            predict = custom_model.predict(df) 
        elif model:
            predict = model.predict(df)
        else:
            print ('Model not good')
            return ('Model is not good')

        return jsonify({'prediction': str(predict)})
    except:
        return jsonify({'trace': traceback.format_exc()})
        

@app.route('/train', methods=['POST'])
def train():
    try:
        json_ = request.json
        user_id = json_[0]['account_id']

        df = pd.DataFrame(json_)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['weekday'] = df['date'].dt.dayofweek
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df = df.dropna(subset=['amount','date'])
        df = df.drop(['date'], axis=1)
        df = df[df['type'] == 1]
        x = df.drop('amount', axis=1)
        y = df['amount']

        new_custom_model = RandomForestRegressor(max_depth=3, n_estimators=500)
        new_custom_model.fit(x, y)
        
        joblib.dump(model, 'custom_models/' + str(user_id) + '_model.pkl')

        return jsonify({'success': True})
    except:
        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345 
    
    model = joblib.load('model.pkl') 
    print ('Model loaded')
    columns = joblib.load('columns.pkl')
    print ('Model columns loaded')
    app.run(port=port, debug=True)