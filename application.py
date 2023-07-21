from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Import ridge regressor model and standard scaler pickle
linreg_model = pickle.load(open('models/linreg.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


# Route for home page
#@app.route("/")
##   return render_template('index.html')
@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Item_Weight = float(request.form.get('Item_Weight'))
        Item_Fat_Content = float(request.form.get('Item_Fat_Content'))
        Item_Visibility=float(request.form.get('Item_Visibility'))
        Item_Type=float(request.form.get('Item_Type'))
        Item_MRP=float(request.form.get('Item_MRP'))
        Outlet_Identifier=float(request.form.get('Outlet_Identifier'))
        Outlet_Establishment_Year=float(request.form.get('Outlet_Establishment_Year'))
        Outlet_Size=float(request.form. get('Outlet_Size'))
        Outlet_Location_Type=float(request.form.get('Outlet_Location_Type')) 
        Outlet_Type=float(request.form.get('Outlet_Type')) 

        new_data_scaled = standard_scaler.transform([[Item_Weight, Item_Fat_Content, Item_Visibility,Item_Type,
        Item_MRP, Outlet_Identifier, Outlet_Establishment_Year,
        Outlet_Size, Outlet_Location_Type, Outlet_Type]])
        result = linreg_model.predict(new_data_scaled)

        return render_template('home.html',result = result[0])
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
