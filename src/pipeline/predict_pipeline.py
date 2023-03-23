import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

from sklearn.preprocessing import StandardScaler


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,


        Delivery_person_Age: float,
        Delivery_person_Ratings: float,
        Restaurant_latitude:float,
        Restaurant_longitude:float,
        Delivery_location_latitude:float,
        Delivery_location_longitude:float,
        Order_Date:str,
        Time_Orderd:str,
        Time_Order_picked: str,
        Weather_conditions: str,
        Road_traffic_density: str,
        Vehicle_condition: int,
        Type_of_order: str,
        Type_of_vehicle: str,
        multiple_deliveries: int,
        Festival: str,
        City: str
        
        
        ):

        self.Delivery_person_Age = Delivery_person_Age

        self.Delivery_person_Ratings = Delivery_person_Ratings

        self.Restaurant_latitude = Restaurant_latitude

        self.Restaurant_longitude = Restaurant_longitude

        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude

        self.Order_Date = Order_Date

        self.Time_Orderd = Time_Orderd
        self.Time_Order_picked = Time_Order_picked
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Delivery_person_Age": [self.Delivery_person_Age],
                "Delivery_person_Ratings": [self.Delivery_person_Ratings],
                "Restaurant_latitude": [self.Restaurant_latitude],
                "Restaurant_longitude": [self.Restaurant_longitude],
                "Delivery_location_latitude": [self.Delivery_location_latitude],
                "Delivery_location_longitude": [self.Delivery_location_longitude],
                "Order_Date": [self.Order_Date],
                "Time_Orderd": [self.Time_Orderd],
                "Time_Order_picked": [self.Time_Order_picked],
                "Weather_conditions": [self.Weather_conditions],
                "Road_traffic_density": [self.Road_traffic_density],
                "Vehicle_condition": [self.Vehicle_condition],
                "Type_of_order": [self.Type_of_order],
                "Type_of_vehicle": [self.Type_of_vehicle],
                "multiple_deliveries": [self.multiple_deliveries],
                "Festival": [self.Festival],
                "City": [self.City],
         
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

