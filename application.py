from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Delivery_person_Age=request.form.get('Delivery_person_Age'),
            Delivery_person_Ratings=request.form.get('Delivery_person_Ratings'),
            Restaurant_latitude=request.form.get('Restaurant_latitude'),
            Restaurant_longitude=request.form.get('Restaurant_longitude'),
            Delivery_location_latitude=request.form.get('Delivery_location_latitude'),
            Delivery_location_longitude=request.form.get('Delivery_location_longitude'),
            Order_Date=request.form.get('Order_Date'),
            Time_Orderd=request.form.get('Time_Orderd'),
            Time_Order_picked=request.form.get('Time_Order_picked'),
            Weather_conditions=request.form.get('Weather_conditions'),
            Road_traffic_density=request.form.get('Road_traffic_density'),
            Type_of_order=request.form.get('Type_of_order'),
            Type_of_vehicle=request.form.get('Type_of_vehicle'),
            multiple_deliveries=request.form.get('multiple_deliveries'),
            Festival=request.form.get('Festival'),
            City=request.form.get('City')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        


