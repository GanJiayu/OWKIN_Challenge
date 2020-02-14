## Data challenge OWKIN Report 

### submit by GAN Jiayu

### 1. Data
#### 1.1. Sample size: 300, split into 250 in training set and 50 in validation set.
#### 1.2. Features: For each patient, we have clinic data, radiomic data and segmented CT scan. 
#### 1.3. Clinic data: Large-granularity histology type and TNM-stage. No need to convert histology type into one-hot coding since XGBoost model is able to process one-value category feature.
#### 1.4. Radiomic data: highly correlated, can be observed from heatmap of correlation matrix. Use PCA to reduce the dimension of radiomic data.
#### 1.5. Image data: CT+Segmentation.

### 2. Target
#### 2.1. Objective: Predict survival time of the patient and the event (local_recurrent/metastasis) upon death.
#### 2.2. Metric: Survival C-index, note that this function is not differentiable, so I still use RMSE/MSE as loss function when training model.

### 3. Preprocessing
#### 3.1. Clinic string -> tokenize
#### 3.2. Radiomic data -> PCA (10 components to recover 95% variance)
#### 3.3. Clinic + Radiomic -> Imputer (missing values in age) + normalizer
#### 3.4. CT image -> (pixel / 1024.) + 1

### 4. Model
The following approaches were implemented with a few fine-tuning trials (Not enough time) 

#### 4.1. 
    A XGBoost model takes clinic and PCA-radiomic features as input and predicts survival time (C-index: 0.66 evaluated on validation set)

#### 4.2. 
    Stacked XGBoost, the first one takes clinic and PCA-radiomic features as input and predicts event upon death (0/1). The successive one takes clinic, radiomic and event feature as input. (C-index 0.67 evaluated on validation set)

#### 4.3. 
    A CNN model takes segmented CT as input and predicts survival time.
    The architecture is derived from REF: https://arxiv.org/ftp/arxiv/papers/1812/1812.00291.pdf 
    (C-index 0.73 evaluated on validation set)

#### 4.4. 
    A CNN model takes segmented CT and clinic/radiomic data and predicts survival time. clinic/radiomic data were injected into the 2nd fully connected layer.
    (C-index 0.66 evaluated on validation set)

#### 4.5. 
    Take the CNN model trained in (4.3) as a feature extractor, the high-level abstract features were taken from output of the last fully connect layer. One successive XGBoost takes clinic, radiomic and convoluted image features as input.
    (C-index 0.73 evaluated on validation set)

### 5. Other tools
#### 5.1.
    RFECV - feature selection based on performance evaluated on CV.
#### 5.2.
    RandomizedSearchCV - hyperparameter fine-tuning tool for XGBoost model.

