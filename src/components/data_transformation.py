import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from category_encoders.target_encoder import TargetEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['Delivery_person_Age','Delivery_person_Ratings','multiple_deliveries','Vehicle_condition']

            categorical_columns_with_less_classes=['Weather_conditions','Road_traffic_density',
                                       'Type_of_order','Type_of_vehicle','Festival','City']

            categorical_columns_with_multiple_classes=['Order_Date','Time_Orderd','Time_Order_picked']


            num_pipeline= Pipeline(
                steps=[

                ("imputer",SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline_with_less_classes=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )
            cat_pipeline_with_multiple_classes=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("TargetEncoder",TargetEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns with less classes: {cat_pipeline_with_less_classes}")
            logging.info(f"Categorical columns with multiple classes: {cat_pipeline_with_multiple_classes}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                
                    
                
                ("cat_pipeline_with_less_classes",cat_pipeline_with_less_classes,categorical_columns_with_less_classes),
                ("cat_pipeline_with_multiple_classes",cat_pipeline_with_multiple_classes,categorical_columns_with_multiple_classes)
                    
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        



        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Time_taken (min)"
            numerical_columns = ['Delivery_person_Age','Delivery_person_Ratings','multiple_deliveries','Vehicle_condition']


            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df,target_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
