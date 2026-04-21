# Data Quality & Known Issues Report

This document tracks data issues discovered during the Exploratory Data Analysis (EDA) phase.

## 1. Missing Values
The datasets contain missing values that will need to be handled via imputation or dropping before model training.

### Main Dataset (`Chronic_Kidney_Dsease_data.csv`)
No missing values detected in the main dataset.

### Training Dataset (`kidney_disease_train.csv`)
|      |   Missing Count |
|:-----|----------------:|
| rbc  |             107 |
| rc   |              93 |
| wc   |              77 |
| pot  |              68 |
| sod  |              67 |
| pcv  |              51 |
| pc   |              50 |
| hemo |              39 |
| su   |              38 |
| sg   |              36 |
| al   |              35 |
| bgr  |              33 |
| bu   |              14 |
| sc   |              12 |
| bp   |               9 |
| age  |               5 |
| ba   |               4 |
| pcc  |               4 |
| htn  |               1 |
| dm   |               1 |
| cad  |               1 |

## 2. Target Distribution (Class Imbalance)
Monitoring the distribution of the target variable to decide if techniques like SMOTE or class weighting are necessary.

### Main Dataset Target (`Diagnosis`)
|   Diagnosis |   Percentage (%) |
|------------:|-----------------:|
|           1 |            91.86 |
|           0 |             8.14 |

### Training Dataset Target (`classification`)
| classification   |   Percentage (%) |
|:-----------------|-----------------:|
| ckd              |            62.14 |
| notckd           |            37.86 |

## 3. Data Type & Formatting Issues
* Note: Some columns in the training dataset may contain string representations of missing values (e.g., `?` or `	?`) or incorrect types that will need converting to numeric.
* Patient IDs should be dropped prior to training to prevent data leakage.
