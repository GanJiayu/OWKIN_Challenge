import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

def load_clinic_data(fpath_train, fpath_test):
    x_train_c = pd.read_csv(fpath_train)
    x_test_c = pd.read_csv(fpath_test)
    
    # HISTOLOGY
    
    _, hist_keys = pd.factorize(x_train_c['Histology'], na_sentinel=True)
    #print(hist_keys)
    hist_codes = [1, 2, 3, 4, 1, 3, 4]
    hist_LUT = dict(zip(hist_keys, hist_codes))

    # Convert by Look-up table
    def Transform_LUT(x, LUT=hist_LUT):
        # is NAN
        if type(x) == np.float:
            return 0
        # string
        else:
            return LUT[x]

    x_train_c['Histology'] = x_train_c['Histology'].apply(Transform_LUT)
    x_test_c['Histology'] = x_test_c['Histology'].apply(Transform_LUT)
    
    # Dataset
    _, dataset_keys = pd.factorize(x_train_c['SourceDataset'].str.lower())
    dataset_codes = [0, 1]
    dataset_LUT = dict(zip(dataset_keys, dataset_codes))
    
    x_train_c['SourceDataset'] = x_train_c['SourceDataset'].apply(lambda x: dataset_LUT[x])
    x_test_c['SourceDataset'] = x_test_c['SourceDataset'].apply(lambda x: dataset_LUT[x])
    
    
    x_train_c = x_train_c.set_index('PatientID')
    x_test_c = x_test_c.set_index('PatientID')
    
    return x_train_c, x_test_c, hist_LUT, dataset_LUT

def load_radiomic_data(fpath_train, fpath_test):
    x_train_r = pd.read_csv(fpath_train, skiprows=[1,2])
    x_test_r = pd.read_csv(fpath_test, skiprows=[1,2])
    
    x_train_r = x_train_r.rename(
        columns={'Unnamed: 0': 'PatientID'})
    
    x_test_r = x_test_r.rename(
        columns={'Unnamed: 0': 'PatientID'})
    
    x_train_r.set_index('PatientID')
    x_test_r.set_index('PatientID')

    return x_train_r, x_test_r

def load_label_data(fpath):
    return pd.read_csv(fpath)

def join_train_data(df_c, df_r, df_y):
    X_train = pd.merge(
            df_c, df_r, how='left',
            left_on=['PatientID'],
            right_on=['PatientID'],
            sort=False)
    XY_train = pd.merge(
            X_train, df_y, how='left',
            left_on=['PatientID'],
            right_on=['PatientID'],
            sort=False)
    XY_train = XY_train.drop('PatientID', axis=1)
    return XY_train
    
def load_test_data(fpath_c, fpath_r, lut1, lut2):
    x_test_c = pd.read_csv(fpath_c)
    x_test_r = pd.read_csv(fpath_r, skiprows=[1,2])
    # Convert by Look-up table
    def Transform_LUT(x, LUT=lut1):
        # is NAN
        if type(x) == np.float:
            return 0
        # string
        else:
            return LUT[x]

    x_test_c['Histology'] = x_test_c['Histology'].apply(Transform_LUT)

    x_test_c['SourceDataset'] = x_test_c['SourceDataset'].apply(lambda x: lut2[x])
    
    x_test_c = x_test_c.set_index('PatientID')
    
    x_test_r = x_test_r.rename(
        columns={'Unnamed: 0': 'PatientID'})
    
    x_test_r.set_index('PatientID')

    X_test = pd.merge(
                x_test_c, x_test_r, how='left',
                left_on=['PatientID'],
                right_on=['PatientID'],
                sort=False)

    X_test_ID = X_test['PatientID'].values
    X_test = X_test.drop('PatientID', axis=1)
    return X_test, X_test_ID

def prepare_train_data(df, feature_List=None, normalize=True, threshold=None):
    X = df.drop(['SurvivalTime','Event'], axis=1)
    Y = df['SurvivalTime'].values
    Y2 = df['Event'].values

    # Feature Selection
    if feature_List is not None:
        X = X[feature_List]
    
    col_name = X.columns
    X = X.values

    # Imputer
    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(X)

    # Normalize
    scr = preprocessing.StandardScaler()
    if normalize:
        X = scr.fit_transform(X)

    if threshold is not None:
        X[X<=-threshold]=-threshold
        X[X>=threshold]=threshold

    return X, Y, Y2, col_name, scr