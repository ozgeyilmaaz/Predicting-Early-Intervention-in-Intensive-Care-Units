# Predicting-Early-Intervention-in-Intensive-Care-Units
Thesis

# SUMMARY
ICU is a costly service with limited resources, where patients with serious health problems and who need to be followed up receive treatment. The general health status of patients in ICU may deteriorate suddenly and deteriorate rapidly. For this reason, these services are equipped with special medical devices, as early and correct interventions are required for patients.
ICU admissions are used to monitor and treat conditions such as organ failure, heart attacks, severe physical injuries, strokes, COPD, sepsis, and coma. 
Data such as vital signs, diagnoses, laboratory data of patients are stored in ICU. These data are used for patient follow-up, disease diagnosis and treatment. Intensive care units have become environments where it is necessary and appropriate to use artificial intelligence and machine learning applications because EMR data is too large and complex to be processed in the human brain. These practices paved the way for more efficient use of both human resources and hospital equipment.
Early intervention is of critical importance in these wards where patient mortality rates are high. However, estimating early intervention is an extremely difficult and complex process. Early interventions are the most important factor in improving the health status of patients and getting rid of the future effects of the health problem.
Artificial intelligence applications have gained importance in intensive care units for purposes such as readmissions, death rates, and early intervention estimation, and these studies have increased.
MIMIC-III dataset which contains information about patients in ICU was used in this study. The information in the data comprises details about patients' experiences at the hospital since they first entered the pertinent data units. The intensive care unit's early interventions were predicted using this information using artificial intelligence and machine learning techniques.

# Problem Definition
Resources in intensive care units are scarce, especially in terms of medical personnel. Due to its superior technological equipment, it presents financial difficulties. Due to the patients' chronic impairment or the high mortality rate, it is a vitally crucial component.
Electronic medical records (EMR) gathered from each patient allows for the collection of a lot of data in this area because of the intensive care units' density and evolving technology. However, it is highly difficult, time-consuming, and possibly impossible for the human mind to comprehend all of this information.
The patients' post-release health status will improve and the length of stay and mortality rates will be reduced as a result of early assessment of the patient's conditions. Because of this, early action is crucial in the intensive care unit, but it is exceedingly challenging for medical professionals to make an early forecast. These estimates in the ICU also make it appear possible to use less resources and lessen the budgetary burden. This estimation may be made achievable through AI studies and ML applications.
In generally, we aim to predict early intervention in intensive care units with machine learning techniques.

# Purpose of Thesis
The aim of this study is to perform the early estimation of emergency interventions using machine learning algorithms in order to reduce the mortality rates of patients hospitalized in the intensive care unit with limited resources and high costs, to provide better service and to use resources more efficiently.

# 1.2 The Databased Used
MIMIC-III is large, single-center, relational database consisting of 26 tables. It includes data on patients who have been admitted to an expansive tertiary hospital's ICU. The data includes the information of the patients during the processes they spent in the hospital as of their entry to the relevant units in the data set. This information includes vital signs, medications, laboratory measurements, observations and notes prepared by caregivers, fluid balance, procedure codes, diagnostic codes, imaging reports, hospital stay, survival data, and more. Data measurement types include demographics, clinical measurements, interventions, billing, medical history dictionary, pharmacotherapy, clinical laboratory tests and medical data. Data technology types are electronic medical record, medical record, electronic billing system, medical coding process document, free text format. Data is obtained with the help of these technologies. In recent years, the use of digital health record systems has increased gradually. 

# Tables and Columns Used in This Study
![image](https://github.com/ozgeyilmaaz/Predicting-Early-Intervention-in-Intensive-Care-Units/assets/79103460/a6376aab-75bb-4111-a057-0c0884b87b4e)

# APPLICATION
Here, after importing the required libraries first, the tables to be used will be read appropriately, combined, and then, after performing the transform and imputation operations on the combined main table, a target column will be created for sepsis as a result of providing/not providing certain disease conditions to the table. After the data set is divided into train and test, the created models will be trained, and then the best model will be decided as a result of comparing the metrics of the models.
## Importing the Necessary Libraries
•	import numpy as np
It enables multidimensional array operations, data analysis and scientific computation.
•	import pandas as pd
It enables reading, analysis, manipulation and merging of data.
•	from sklearn.model_selection import train_test_split
It allows the data to be divided into training and test sets. With this function, some of the dataset is used to train the model, while some is used to test it. It basically takes input and target variables, train size as parameters.
•	from sklearn.linear_model import LogisticRegression
The 'LogisticRegression' class in the 'sklearn.linear_model' module is used to implement logistic regression, a classification algorithm. This class trains a model using input variables and target classes and is then used to classify new input features.
•	from sklearn.tree import DecisionTreeClassifier
The decision tree is used to build classification models. Required parameters are 'criterion' (Determines the split criterion. Usually set to "gini" or "entropy"), 'max_depth' (Determines the maximum depth of the tree), min_samples_split' (Determines the minimum number of samples in the node before splitting), 'min_samples_leaf' (Leaf) specifies the minimum number of samples in leaf nodes), 'max_features' (sets the minimum number of samples in leaf nodes).
•	from tensorflow.keras.models import Sequential
The Sequentials class in the 'tensorflow.keraws.models' module is used to create neural networks via Keras. It provides the possibility to add interconnected layers sequentially.
•	from tensorflow.keras.layers import Dense
The Dense class in the 'tensorflow.keras.layers' module is used to create fully connected layers. These layers connect each output of each neuron in the previous layer to each input of each neuron in the next layer. Fully connected layers are one of the most basic building blocks of a neural network model. The required parameters are the 'units' and 'activation' parameters. 'units' specifies the number of neurons in the layer, while 'activation' specifies the activation function to be used to calculate the layer's output.
•	from tensorflow.keras.layers import Dropout
The Dropout class is an editing technique used to prevent the model from over-learning. This class drops a randomly selected percentage to the output of the layer it takes as input and increases the remaining outputs proportionally. This is done before moving on to the next layer. Its parameter is 'rate'. This parameter specifies the probability of randomly dropping units at the output of the layer.
•	from sklearn.preprocessing import StandardScaler
‘StandardScaler’ is used to standardize the data to avoid overfitting. 
•	from sklearn.metrics import accuracy_score
‘accuracy_score’ is used evaluation metric for classification problems in machine learning. It is used to measure the accuracy or correctness of a classifier's predictions by comparing them to the true labels or targets.
•	from sklearn.metrics import roc_curve, auc
‘roc_curve’ and ‘auc’ are commonly used for evaluating the performance of binary classification models. The ROC curve is a graphical representation of the performance of a classifier at various classification thresholds. The ‘auc’ function calculates the Area Under the ROC Curve (AUROC) for a binary classification model. 
•	from datetime import datetime as dt
It is used to convert the necessary columns into date format in order to be suitable for use in the model.
5.2. Reading the Data Tables
The 'read_csv' function under the 'pandas' library is used to read the data tables. This function takes the path of the file as a parameter. In this step, ADMISSIONS, CHARTEVENTS, ICUSTAYS, LABEVENTS and PATIENTS tables to be used for preprocessing and merge are read and assigned to a variable. D_ITEMS and D_LABITEMS tables were also read for analysis to determine which ID is kept in defining the parameters to be entered into the model. During the file reading, the columns intended to be used in the model are defined in the 'use_cols' parameter in ‘read_csv’ function.

## Preprocessing
First, the ADMISSIONS and PATIENTS tables are combined via SUBJECT_ID. Then, the dummy variables of the GENDER column, which will be input to the model, were found and a gender column was removed from the table. Afterwards, the patient's admission date and birth date columns, which were used to calculate the age on this table, were brought into datetime format and then the age column was added by subtracting each other. Patients younger than 18 years of age were excluded from the data.

In the tables in the data set, the IDs of the patients' medical data items and the values of these data are kept in two separate columns. The appropriate format for the model is to create a column for each medical data. In addition, when the data set was examined, it was observed that there were ids corresponding to more than one different medical data item containing the same description. The ids of which medical data items correspond to the parameters to be used in the models (systolic blood pressure, diastolic blood pressure, blood oxygen saturation, temperature, heart rate, respiratory rate, carbondioxide, white blood cells, pH, age) and the corresponding measurements were kept in a dictionary. This dictionary contains key-value pairs of medical data items. Keys represent names of medical data items, while values are defined as a list containing unique codes corresponding to the relevant medical data item. Temperature is kept in different keys as it has different ITEMIDs than Fahrenheit and Celsius unit.
The CARTEVENTS and LABEVENTS tables that keep the events of the patients are combined one under the other. Using the Dictionary, the ids have been replaced with the corresponding medical data item names. Each medical data item was converted into columns according to their names and their values were placed in these columns.
The Fahrenheit column is converted to celcius. Temperature values below or above certain threshold values were determined as outliers and were excluded from the data.
Patients with a hospital stay of less than 12 hours were excluded from the ICUSTAYS table and then merged with other tables.

Since the different parameter values were measured one after another at close times, the values were kept in one-hour periods. These periods were calculated using the patient's admission time to the hospital and the times when events were held. It has been determined that the measurement times of some events were entered before the patient was admitted to the hospital. These records have been removed from the data.
In order to fill in the parameter values to be used in the model, the data was first grouped on the basis of patient, hospitalization and icustay ids, and then forward fill was performed first. The average in this grouping is entered for the values that may remain after this process. If there are still nan values, the average value was taken by grouping on the basis of patient ID.

Since this study takes into account the examination and intervention of sepsis frequently seen in intensive care units, one of the definition for sepsis will be continued. On the other hand, there is no physiological or anatomical marker that clearly defines sepsis. 

According to Bone et al (1992) sepsis can be defined as suspected infection and two or more of the following conditions: 
•	temperature > 38 °C or < 36 °C 
•	heart rate > 90 beats per minute 
•	respiratory rate > 20 beats per minute
•	white blood cell count > 12,000 cells/μl, < 4000 cells/μl.
A new column is opened for each of these conditions and its value is 1 if they meet the defined criteria, otherwise 0 is entered. A sepsis condition column is created and the value of sepsis condition is assigned as 1 if the sums of each condition are greater than 1, and 0 if not. If this situation continues for 5 hours from the moment the sepsis condition is 1, it takes the value 1 in a new column denoting the onset of sepsis of the patient at that moment.

An attempt was made to predict the onset of sepsis 3, 6, and 12 hours later, using data from the previous 6 hours.

## Splitting the Data Table into Train and Test Set
With the 'train_test_split' function under 'sklearn.model_selection', the data set will be divided into train and test. %80 of the data is train and %20 of the data is test set.  

## Creating the Models to be Used
LSTM, decision tree and logistic regression models created. 
For the LSTM model, the 6 hour timestamp values are kept in another dimension. LSTM layer has 32 neurons and input shape is (6, 19). One dense layer is added and it has 1 neuron and it’s activation function is ‘sigmoid’. ‘binary_crossentropy’ is selected for the loss function and ‘adam’ is selected for the optimizer.

# CONCLUSION
Patients with major medical issues that require follow-up care are treated in the intensive care unit, a pricey service with little resources. Patients in intensive care units may experience sudden and fast deterioration in their general health status. Because patients need prompt and accurate interventions, these services are outfitted with specialized medical equipment.

In order to monitor and treat conditions like organ failure, heart attacks, severe physical injuries like gastrointestinal bleeding, COPD, sepsis patients are admitted to the critical care unit.
Intensive care units have evolved into settings where artificial intelligence and machine learning solutions are required and acceptable. EMR data is too big and complicated for the human brain to analyze. These solutions paved the way for the more effective use of hospital supplies and staff as well.
Estimating early intervention, however, is a very challenging and intricate task. The most crucial element in enhancing patients' condition and eradicating long-term repercussions of a health issue is early intervention.

In this investigation, the MIMIC-III dataset was utilized. The data comprises details about patients' experiences at the hospital since they first entered the pertinent data units. The intensive care unit's early interventions were predicted using this information using artificial intelligence and machine learning techniques.

Artificial intelligence studies have been conducted in intensive care units for the early prediction of sepsis and the following: Normotensive or hypotensive, circulatory failure, Acute Kidney Damage (AKI) in sepsis patients, need for fluid, patients at risk for prolonged mechanical ventilation and tracheostomy insertion, rebleeding in patients with GI bleeding and the potential outcome of a blood culture test.

Machine learning is being used to try and accurately diagnose medical issues, including sepsis. There are three main types of machine learning techniques that can be used to predict sepsis: common classification models, deep neural networks, and ensemble learning techniques. 

Based on the comparison results that can be seen in Table 6.1, it is seen that the model with the highest AUROC (0.85) value in estimating the sepsis condition after 3 hours is LSTM. Looking at the estimation results for 6 and 12 hours later, it was observed that the AUROC values were close to the Decision Tree model. However, the fact that the LSTM model has a structure that can better capture complex relationships and process data over time makes the LSTM model a preferable model for sepsis prediction. However, it can be understood that a well-prepared Decision Tree model can also be used in models that predict sepsis.

On the other hand, the Logistic Regression model has lower AUROC (3-hours: 0.71; 6-hours: 0.70; 12-hours: 0.67) compared to the other two models. For this reason, it has emerged that the Logistic Regression model is less successful than other models in predicting sepsis status. At this point, it can be thought that the Logistic Regression model is insufficient to capture more complex data structures and dynamics over time. For these reasons, it seems that the Logistic Regression model will not be preferred. In addition, it is seen that the success rate of estimation decreases as the estimation time increases.

![image](https://github.com/ozgeyilmaaz/Predicting-Early-Intervention-in-Intensive-Care-Units/assets/79103460/3c73a69c-3214-4483-b462-14d138499c53)

As a result of this study, it was found reasonable to use the LSTM model to predict sepsis status with the MIMIC-III dataset, but the choice of model may vary depending on the dataset characteristics.

In this study, we can think that the application was made on only one dataset (MIMIC-III), and other models could be preferred in different datasets. The LSTM model established in this study may not have these results on a different data set. In order to ensure the general validity of the studies conducted in this area, it may be a solution to try them in different regions. In addition, the data used in studies in this area differ. Some data are multicenter, while others are monocentric. This illustrates how different datasets can generate diverse findings for studies. Future research that will increase the predictive power in intensive care units should have a data set to support these studies.

The results of this study showed that different machine learning models can be successfully used to predict sepsis status using patients' last 6 hours of data. These estimates are expected to provide a number of benefits in intensive care units.

First, these estimates have been shown to be helpful in optimizing bed occupancy rates in intensive care units. Predicting patients' risk of sepsis can help plan appropriate beds for treatment and manage emergencies more effectively. This can increase the efficiency of intensive care units and enable patients to be treated more quickly.

In addition, these estimates are thought to be effective in reducing mortality rates. Early diagnosis of sepsis can expedite the implementation of necessary treatment protocols and help prevent life-threatening conditions. Thus, patients' chances of survival may increase and mortality rates may decrease.

Another benefit of estimates is to reduce readmission rates. Early detection of sepsis can lead to more comprehensive treatment of patients and speed up the recovery process. This, in turn, can reduce the need for rehospitalization of patients and enable more efficient use of health services.

However, it is thought that these estimates may also help prevent permanent damage after discharge from the intensive care unit. Early intervention and treatment can reduce patients' post-sepsis complications and lower the risk of permanent damage. This, in turn, can improve patients' quality of life and prevent long-term health problems.

Finally, it is thought that these estimations may contribute to the more efficient use of high-tech, expensive equipment, laboratory equipment and health personnel used in the intensive care unit. Early detection of patients' sepsis status can prevent unnecessary testing and procedures and enable more efficient use of resources.

In conclusion, this study demonstrated that different machine learning models can be used successfully to predict sepsis status using the last 6 hours of data from patients. These estimates are expected to provide economic and social benefits such as bed occupancy rates, mortality rates, readmission rates, prevention of permanent damage after evacuation and optimization of health services in intensive care units. This study could be an important step to increase the efficiency of intensive care units and improve the health outcomes of patients.
