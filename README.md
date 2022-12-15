
# **Customer Churn Analysis**

Customer churn is a fundamental problem for the telecommunication (Telco) industry. It is defined as the loss of customers moved from one Telco operator to another. If customer churn can be predicted in advance, such as “is this customer going to leave us within the next X months?”, Telco operators can apply business marketing policies to such churn customers to retain and increase the customer base. In particular, given millions of Telco customers, even reducing the 1% churn rate will lead to a significant profit increase.

We can roughly define the Churn analysis as the whole of analytical studies on “a customer”, “a product or service”, and “the probability of abandonment”. We aim to be aware of this situation (even the customer himself may not be aware of this situation) before the customer leaves us (approaching to leave) and then to take some preventive actions.

Telco churn data includes information about a fictitious telecom company that provided home phone and Internet services to 7,043 California customers in the third quarter. Which customers have left, stayed, or signed up for their service shows?

![images](https://user-images.githubusercontent.com/108968831/207868009-136809e4-be5b-4360-8e40-b41cc99fb41c.jpg)


# Business Problem 
It is desirable to develop a machine learning model that can predict customers who will leave the company. You are expected to perform the necessary data analysis and feature engineering steps before developing the model.


# Description 
- customerID - unique value identifying customer
- gender - whether the customer is a male or a female
- SeniorCitizen - whether the customer is a senior citizen or not (1, 0)
- Partner - whether the customer has a partner or not (Yes, No)
- Dependents - whether the customer has dependents or not (Yes, No). A dependent is a person who relies on another as a primary source of income,
- tenure - number of months the customer has stayed with the company
- PhoneService - whether the customer has a phone service or not (Yes, No)
- MultipleLines - whether the customer has multiple lines or not (Yes, No, No phone service)
- InternetService - customer’s internet service provider (DSL, Fiber optic, No)
- OnlineSecurity - whether the customer has online security or not (Yes, No, No internet service)
- OnlineBackup - whether the customer has online backup or not (Yes, No, No internet service)
- DeviceProtection - whether the customer has device protection or not (Yes, No, No internet service)
- TechSupport - whether the customer has tech support or not (Yes, No, No internet service)
- StreamingTV - whether the customer has streaming TV or not (Yes, No, No internet service)
- StreamingMovies - whether the customer has streaming movies or not (Yes, No, No internet service)
- Contract - type of contract according to duration (Month-to-month, One year, Two year)
- PaperlessBilling - bills issued in paperless form (Yes, No)
- PaymentMethod - payment method used by customer (Electronic check, Mailed check, Credit card (automatic), Bank transfer (automatic))
- MonthlyCharges - amount of charge for service on monthly bases
- TotalCharges - cumulative charges for service during subscription (tenure) period
- Churn – output value, predict variable
# Steps To Follow
he various steps involved in the Machine Learning Pipeline are :

- Import Necessary Dependencies :
- Read and Load the Dataset :
- Exploratory Data Analysis & Data Cleaning
- Data Visualization
- Feature Selection
- Feature Engineering
- Splitting dataset into train and test datasets
- Model Evaluation
- Hyperparameter tuning
- Model Deployment
- Conclusion
## Step 1 :
## Step 2 : Read and Load the Dataset
We now know that we are working with a typical CSV file (i.e., the delimiter is ,, etc.). We proceed to loading the data into memory.
## Step 3 : Exploratory Data Analysis & Data Cleaning
- Inspect dataset
- Columns/features in data
- Length of the dataset
- Shape of data
- Data information
- Inspect Unique Values in each Columns
- Checking for Null values
- Check total number of missing values in columns
- Dimensional Reduction : part 1
- Creating Independent and Dependent variable
## Step 4 : Data Visualization
- Inspecting target column
- Visualize categorical features relationship with target column
- Visualize numerical features relationship with target column
- Visualize Outilers
## Step 5 : Feature Selection
Feature selection is the process of reducing the number of input variables when developing a predictive model.

It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model.

- **Mutual Information** for feature selection is used.
It measures the entropy drops under the condition of the target value.

The MI score will fall in the range from 0 to ∞. The higher value, the closer connection between this feature and the target, which suggests that we should put this feature in the training dataset. If the MI score is 0 the low score suggests a weak connection between this feature and the target.

- Visualize Features Importance
- Dimesional Redection : Part 2
## Step 6 : Feature Engineering
Feature engineering is a machine learning technique that leverages data to create new variables that aren’t in the training set. It can produce new features for both supervised and unsupervised learning, with the goal of simplifying and speeding up data transformations while also enhancing model accuracy. Feature engineering is required when working with machine learning models.


- **Label Encoding** is used to replace categorical values with numerical values. This encoding replaces every category with a numerical label.
- **One-Hot Encoder** is straight but it has the disadvantage that the numeric values can be misinterpreted by algorithms as having some sort of order in them. This ordering issue is addressed in another common alternative approach called ‘One-Hot Encoding’. In this strategy, each category value is converted into a new column and assigned a 1 or 0 (notation for true/false) value to the column.
- **Checking target incidence** for Imbalanced data handling , Imbalanced data refers to those types of datasets where the target class has an uneven distribution of observations, i.e one class label has a very high number of observations and the other has a very low number of observations. 
- **Imbalanced data handling** : Imbalanced classification involves developing predictive models on classification datasets that have a severe class imbalance.
  
  The challenge of working with imbalanced datasets is that most machine learning techniques will ignore, and in turn have poor performance on, the minority class, although typically it is performance on the minority class that is most important.
  
  One approach to addressing imbalanced datasets is to oversample the minority class. The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. Instead, new examples can be synthesized from the existing examples. This is a type of data augmentation for the minority class and is referred to as the Synthetic Minority Oversampling Technique, or SMOTE for short.
## Step 7 : Data Normalization
It is a common practice in machine learning which consists of transforming numeric columns to a common scale. In machine learning, some feature values differ from others multiple times. The features with higher values will dominate the learning process; however, it does not mean those variables are more important to predict the target. Data normalization transforms multiscaled data to the same scale. After normalization, all variables have a similar influence on the model, improving the stability and performance of the learning algorithm.
## Step 8 : Model Evaluation
Algorithm selection is a key challenge in any machine learning project since there is not an algorithm that is the best across all projects. Generally, we need to evaluate a set of potential candidates and select for further evaluation those that provide better performance.

After Evalution Gradient boosting classifier gives overall better performance in Crossvalidation, accuracy and roc-auc too.

we will only further evaluate the model that presents higher accuracy using the default hyperparameters. As shown above, this corresponds to the gradient boosting model which shows an CV Score of nearly 85.4%.

## Step 9 : Hyperparameter tuning
In the grid search method, we create a grid of possible values for hyperparameters. Each iteration tries a combination of hyperparameters in a specific order. It fits the model on each and every combination of hyperparameters possible and records the model performance. Finally, it returns the best model with the best hyperparameters.
## Step 10 : Model Deployment
After Evalution Gradient boosting classifier gives overall better performance in Crossvalidation, accuracy and roc-auc too.

![Capture](https://user-images.githubusercontent.com/108968831/207867839-afa092cf-84ba-4da9-91dc-62f831b5db00.JPG)


**GradientBoostingClassifier(max_depth=6, min_samples_split=3, n_estimators=80)**
# **Conculsion**

we have walked through Telco customer Churn dataset. We started by cleaning the data , dimensional reduction and analyzing it with visualization.
In a next step - Feature extraction, feature engineering and noramalization od datset for better accuracy and avoid overfiting.
After transforming the data, we tried 9 different machine learning algorithms using default parameters.Here we can conclude Gradient Bossting Classifier Giver overall good Cross validation score and model overall accuracy.
After Hyper parameter , model performance slightly increase approx 85.166 percent .
And also imporve roc-auc score after tuning the model.
