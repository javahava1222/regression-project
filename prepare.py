import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer


def remove_outliers(df, k, col_list):
    ''' remove outliers from an acquired dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([0.25, 0.75])
        
        iqr = q3 - q1
        
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr 
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def add_baseline(train, validate, test):
    ''' Takes train, validate, and test datasets and output baseline'''
    baseline = train.tax_value.mean()
    train['baseline'] = baseline
    validate['baseline'] = baseline
    test['baseline'] = baseline
    return train, validate, test



def wrangle_zillow(df):
    ''' Prepare zillow dataset, remove outliers, drop null value, and clean data  '''
    #remove outlier
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value'])
    #drop null and duplicates
    df = df.dropna().drop_duplicates()
    #change the fips code to counties
    df['location'] = df['fips'].replace({6037.0:'LA', 6059.0: 'Orange', 6111.0:'Ventura'})
    # create columns for the age of houses
    df['age'] = 2017 - df['year_built']

    #change the fips to object
    df.fips = df.fips.astype(object)
    #change the year built to object
    df.year_built = df.year_built.astype(object)

    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    train, validate, test = add_baseline(train, validate, test)       
    
    return train, validate, test    

def scale_zillow(train, validate, test):
    '''
    Takes train, validate, test datasets, make copy, and returns the scaled datasets
    for columns_to_scale.
    '''
    columns_to_scale = ['bedrooms', 'bathrooms', 'tax_value', 'area']

    # Make copy
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    ## MinMaxScaler
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit scaler to data
    scaler.fit(train[columns_to_scale])

    # transform
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    return train_scaled, validate_scaled, test_scaled