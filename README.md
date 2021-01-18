# Glass Type Classification

This project is part of Udacity Capstone Project.  It is performed using two models:

1. Automated ML and
2. Hyperparameters are tuned using HyperDrive.

## Project Set Up and Installation

The project is carried out using below steps

- Import the External dataset
- Train Auto ML model
- Train Hyperdrive model
- Compare model performance
- Deploy best model
- Test model endpoint

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/Flowchart.jpg)

## Dataset

### Overview

Dataset is downloaded from [Kaggle](https://www.kaggle.com/uciml/glass) datasets and also through [UCI ML](https://archive.ics.uci.edu/ml/datasets/Glass+Identification) repository. The same can be uploaded to github and used through Github raw content.

**Citation**:

Creator:

B. German
Central Research Establishment
Home Office Forensic Science Service
Aldermaston, Reading, Berkshire RG7 4PN

Donor:

Vina Spiehler, Ph.D., DABFT
Diagnostic Products Corporation
(213) 776-0180 (ext 3014)

Relevant Papers:

Ian W. Evett and Ernest J. Spiehler. Rule Induction in Forensic Science. Central Research Establishment. Home Office Forensic Science Service. Aldermaston, Reading, Berkshire RG7 4PN
[Web Link]

## Attributes Description

1. Id number: 1 to 214
2. RI: refractive index
3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
4. Mg: Magnesium
5. Al: Aluminum
6. Si: Silicon
7. K: Potassium
8. Ca: Calcium
9. Ba: Barium
10. Fe: Iron
11. Type of glass: (class attribute)
- 1 building_windows_float_processed
- 2 building_windows_non_float_processed
- 3 vehicle_windows_float_processed
- 4 vehicle_windows_non_float_processed (none in this database)
- 5 containers
- 6 tableware
- 7 headlamps

### Task

In this Project we will classify the given dataset into different types based on the percentage of the chemicals.

#### Motivation of collection of dataset

The study of classification of types of glass was motivated by criminological investigation. At the scene of the crime, the glass left can be used as evidence...if it is correctly identified!

### Access

To access the dataset to workspace we have two methods

1: Importing the dataset through local files
2: Importing the dataset through github raw link

Below is the screenshot after the dataset is registered through local files

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/Dataset.JPG)

Accessing the dataset through [Github](https://raw.githubusercontent.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/main/glass.csv)

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/registering_ds.JPG)

- Before proceeding with the implementation we will create a Compute instance to run the Jupyter notebooks

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/Compute%20instance.JPG)

## Hyper parameter tuning using hyperdrive 

Hyperdrive architecture is as below:-

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/Hyperdrive_design.JPG)

- The model I have choosen is Logistic regression with inverse regularisation parameter (--C) which helps in controlling the model from overfitting and (max-iter )- defines the  number of iterations as another hyper-parameter.
- I have used Randomsampling method in the parameter sampling
- I have used Banditpolicy as the early stopping criteria. The modelling stops if the policy doesn't meet the slack factor, delay interval in their prescribed limits/values.

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/HD_config.JPG)

### Run widgets

- Run Details from the Jupyter notebook

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_run_nb.JPG)

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_scatter.JPG)

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_acc_nb.JPG)

- Run details from the workspace

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_run.JPG)

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_models.JPG)

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_child_runs.JPG)

### Best model 

The best hyperdrive model

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_bestrun_page.JPG)

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_bestrun_metrics.JPG)

### Results

Best run id:- 
![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_bestrun_i.JPG)

Besr run parameters :-

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/hd_params.JPG)

## Auto ML

Automl is also known as Automated ML which helps in rapidly performing multiple iteration on different algorithms. It also supports Ensemble methods. Here we get voting ensemble as our best run

The below are the 'Automl' settings and configuration taken

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/Automl%20config.JPG)

- Here the problem is a classification problem hence we have taken task ='classification'
- We have given early timeout to be 30 mins
- n cross validation to be 5
- Compute target is the cluster on which the computation to be performed.

### Run widgets

- Run details from Jupyter notebook

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/aml_run_nb.JPG)

- Run details from Workspace

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/Automl_run.JPG)

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/aml_models.JPG)

## Best model

Here we got **voting classifier** as out best model

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/Aml_bestmodel.JPG)

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/aml_metrics_bestmodel.JPG)

The following illustrates the Best model's run id and best run's parameters

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/Aml_best_run_id.JPG)

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/automl_best_params.JPG)

### Registering the model

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/aml_register.JPG)

## Results

On comparing the best models of both Hyperdrive and Automl , we know that we found Voting classifier through Automl is having better performance

- Hyperdrvie model:-  Accuracy- 0.6615 , AUC_weighted = 0.78
- AutoML model :-     Accuracy- 0.75760 , AUC_weighted = 0.94213

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/voting_run_metrics.JPG)

## Model deployment

Since we found that Automl model is the best model we are going ahead and deploying the same.

We deploy the best model using ACI instance and have also enabled the insights and enabled the Authentication to consume the restpoint.

Procedure to deploy the model

1. Provide a scoring script which will be invoked by the web service call (using scoring.py). 
   The scoring script must have two required functions, init() - that loads your model and run() it runs the obtained model on your input data.
   Then provide the environment files to inference config along with the script file and deploy the service. 
   
2. We can deploy the services from the Azure workspace by configuring the Authentication and then going ahead and enabling the Application insights using log.py file

- Service status is **Healthy**

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/endpoint.JPG)

- Applications Insights is enabled

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/appinsights.JPG)


### Consuming endpoints

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/updating%20endpoint.JPG)

### Testing endpoints

![](https://github.com/Aishwaryasasanapuri/Azure-ML-Capstone-Project/blob/main/screenshots/testing%20service.JPG)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. [Link](https://youtu.be/6MLM2LC9qO8)

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*(Optional):* This is where you can provide information about any standout suggestions that you have attempted.

- We have enabled Application insights

## Future Improvements

- More data would help in getting more insights from the Automl and hyperdrive methods
- Feature engineering can be performed
- Different feature reduction techniques could be used like PCA, RFE 
- We can work on class imbalance problem
- Using Cross validation techniques would help in cribbing problems like overfitting
- Th model can be converted to ONXX format and be deployed on Edge services.

## References :

- [Github starter code](https://github.com/udacity/nd00333-capstone/tree/master/starter_file)
- [How to deploy](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?view=iotedge-2018-06&tabs=python)
- [Enabling insights](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-app-insights)
- [Deploying AZ ML on IoT Edge](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-deploy-machine-learning?context=azure%2Fmachine-learning%2Fservice%2Fcontext%2Fml-context&view=iotedge-2018-06&viewFallbackFrom=azure-ml-py#create-and-deploy-azure-machine-learning-module)
- [ONXX in Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx)

#### Documents refered to fix Error cause during Model deployment
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-visual-studio-code#recommendations-for-error-fix
- https://stackoverflow.com/questions/60391230/azure-504-deploymenttimedout-error-service-deployment-polling-reached-non-succ
- https://github.com/MicrosoftDocs/azure-docs/issues/44806
- https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml#imbalance
- https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice(class)?view=azure-ml-py
