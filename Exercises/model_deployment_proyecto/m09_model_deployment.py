#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
import category_encoders as ce

def predict_price(Year,Mileage,State,Make,Model):

    clf = joblib.load(os.path.dirname(__file__) + '/car_price_prediction.pkl') 
    encoder=joblib.load(os.path.dirname(__file__) + '/encoder.pkl') 
    
    columnas= ["Year","Mileage","State","Make","Model"]
    data=[Year,Mileage,State,Make,Model]
    df = pd.DataFrame(columns=columnas)
    df.loc[len(df)] = data
        
    X_cat = df[["State", "Make", "Model"]]
    X_num = df[["Mileage","Year"]]
    
    
    
    # Create features
    X_cat = encoder.transform(X_cat)
    X = pd.concat([X_num,X_cat],  axis=1, sort = False)

    # Make prediction
    p1 = clf.predict(X)

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add Parameters')
        
    else:

        df = sys.argv[1]

        p1 = predict_price(Year,Mileage,State,Make,Model)
        
        print('Year:',Year,' Mileage:',Mileage,' State:',State,' Make:',Make,' Model:', Model)
        print('Estimated price: ', p1)
        