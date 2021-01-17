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

![](flowchart.jpg)

## Dataset

### Overview

Dataset is downloaded from [Kaggle](https://www.kaggle.com/uciml/glass) datasets and also through [UCI ML](https://archive.ics.uci.edu/ml/datasets/Glass+Identification) repository. The same can be uploaded to github and used through Githubraw content.

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

![](Dataset)

Accessing the dataset through Github

![](registering_ds)

- Before proceeding with the implementation we will create a Compute instance to run the Jupyter notebooks

![](Computer instance)

## Hyper parameter tuning using hyperdrive 

- The model I have choosen is Logistic regression with inverse regularisation parameter (--C) which helps in controlling the model from overfitting and max-iter - number of iterations as another hyper-parameter.
- I have used Randomsampling method in the parameter sampling
- I have used Banditpolicy as the early stopping criteria. The modelling stops if the policy doesn't meet the slack factor, delay interval in their prescribed limits/values.

![]( hyperdrive_config)

### Run widgets

![](hd_runs)

### Best model 

![](hd_bestmodel)

### Results

![](hd_params)
![]( hd_best_i)

## Auto ML

Automl is also known as Automated ML which helps in rapidly performing multiple iteration on different algorithms. It also supports Ensemble methods. Here we get voting ensemble as our best run

The below are the 'Automl' settings and configuration taken

![](automl_config)

- Here the problem is a classification problem hence we have taken task ='classification'
- We have given early timeout to be 30 mins
- n cross validation to be 5
- Compute target is the cluster on which the computation to be performed.

## Best model

Here we got **voting classifier** as out best model


## Results

On comparing the best models of both Hyperdrive and Automl , we know that we found Voting classifier through Automl is having better performance

- Hyperdrvie model:-  Accuracy- 0.66 , AUC_weighted = 0.
- AutoML model :-     Accuracy- 0.66 , AUC_weighted = 0.

![](voting classifier)

## Model deployment

We deploy the best model using ACI instance and have also enabled the insights and enabled the Authentication to consume the restpoint.

![](endpoint)
![](application insights)

### Consuming endpoints

![](updating endpoint)

### Testing endpoints

![](testing service)

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

[Github starter code](https://github.com/udacity/nd00333-capstone/tree/master/starter_file)
[How to deploy](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?view=iotedge-2018-06&tabs=python)
[Enabling insights](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-app-insights)
[Deploying AZ ML on IoT Edge](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-deploy-machine-learning?context=azure%2Fmachine-learning%2Fservice%2Fcontext%2Fml-context&view=iotedge-2018-06&viewFallbackFrom=azure-ml-py#create-and-deploy-azure-machine-learning-module)

#### Error fixes
[](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-visual-studio-code#recommendations-for-error-fix)
[](https://stackoverflow.com/questions/60391230/azure-504-deploymenttimedout-error-service-deployment-polling-reached-non-succ)
[](https://github.com/MicrosoftDocs/azure-docs/issues/44806)
https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml#imbalance
https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice(class)?view=azure-ml-py
