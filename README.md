# Forecasting-Backorders-In-Inventory
This is an end to end machine learning system for predicting the backorders based on the training data.

## Problem statement
An order placed with a supplier for a product that is momentarily out of stock but may ensure delivery of the requested goods or services by a specific date in the future, since the production of the products is already begun. A backorder shows that there is a gap between the supply of a certain good or service and the demand for it. But by anticipating which things will be back-ordered, planning can be streamlined at several levels, preventing unexpected strain on production, logistics, and transportation.

## Solution proposed
ERP systems generate a lot of data (mainly structured) and also contain a lot of historical data. If this data can be properly utilized, a predictive model to forecast backorders can be constructed. Based on past data from inventory, supply chain, and sales. Now the task is a binary Classification where we have to predict if our product goes on backorder or not.\
Yes: If the product goes on backorder.\
NO: If the product is not on backorder. 

The predictions will be done on the new batches obtained for every scheduled interval of time and store those predictions in the s3 bucket for later usage. Airflow was used for orhestrating the training and prediction pipelines.

For a reference, single instance prediction is also implemented.

## Tech Stack Used
1. Python
2. Sklearn for machine learning algorithms
3. Flask for creating an web application
4. Airflow is used for pipeline(train and prediction) orchestration.
5. MongoDB Atlas for database operations
6. Docker is used for container builds
7. Terraform used for managing the Infrastructure 
8. Github actions for implementing CI/CD

## Infrastructure Required.
1. AWS EC2 instances for deploying the app
2. AWS S3 Buckets for data storage, feature store and artifacts store
3. AWS ECR for storing the container images

## Dashboarding
1. Grafana is used for metrics visulization
2. Prometheus
3. Node Exporter
4. Promtail
5. Loki



project demo: https://youtu.be/UIrTIteZvwI

![](https://ml-ops.org/img/mlops-phasen.jpg)
## Steps to run project in local system
1. Build docker image
```
docker build -t bo:lts .
```

2. Set envment variable
```
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export MONGO_DB_URL=
export AWS_DEFAULT_REGION=
export IMAGE_NAME= "bo:lts"
```
3. setup airflow in local

4. To start your application
```
docker-compose up
```
5. To stop your application
```
docker-compose down
``` 



AIRFLOW SETUP

## How to setup airflow

Set airflow directory
```
export AIRFLOW_HOME="/home/anil/Forecasting-Backorders-In-Inventory/airflow"
```

To install airflow 
```
pip install apache-airflow
```

To configure databse
```
airflow db init
```

To create login user for airflow
```
airflow users create  -e anilsai029@gmail.com -f anil -l sai -p admin -r Admin  -u admin
```
To start scheduler
```
airflow scheduler
```
To launch airflow server
```
airflow webserver -p <port_number>
```

Update in airflow.cfg
```
enable_xcom_pickling = True
```
## UI for single Instance prediction
![](https://github.com/anilans029/Forecasting-Backorders-In-Inventory/blob/main/Documents/UI.png?raw=true)

![](https://github.com/anilans029/Forecasting-Backorders-In-Inventory/blob/main/Documents/result.png?raw=true)

## system monitoring
![](https://github.com/anilans029/Forecasting-Backorders-In-Inventory/blob/main/Documents/grafana_system_monitoring.png?raw=true)