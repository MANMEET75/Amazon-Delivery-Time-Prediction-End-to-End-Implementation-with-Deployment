# Amazon Business Research Analyst Problem Statment
I solved a use case for Amazon Business Research Analyst by predicting the delivery time for an order. I created three environments: research, development, and production. In the research environment, I collected and cleaned data from Kaggle, performed exploratory data analysis, feature engineering, feature selection, and model building. In the development environment, I converted the code into modular code and created a web application using Flask. In the production environment, I deployed the model using Amazon Elastic Beanstalk and created a code pipeline for continuous delivery.


## Research environment
In the research environment, I collected and cleaned data from Kaggle, performed exploratory data analysis, feature engineering, feature selection, and model building using regression algorithms with hyperparameter optimization for maximum accuracy. Open source libraries like category encoder and feature engine were used for reproducibility and robustness.

## Development environment
In the development environment, we convert our code into modular code and create a web application using Flask. We create a repository on GitHub, create a virtual environment using anaconda, and create multiple files for components like data ingestion, data transformation, model building, and prediction pipeline. We also build an interactive UI for the ML application using full stack web development skills.


## Production environment
Deployed model using Amazon Elastic Beanstalk with .ebextension configuration file. Deployment process includes creating new applications on AWS and integrating GitHub repository with AWS instance through CodePipeline. Continous delivery feature allows for easy updates to the codebase on GitHub.
