# My Functions for data Science :) 
# imports 
import pandas as pd
import numpy as np



# This function will output descriptive statistics of the data

def df_summary(df):

#     prints the descriptive statistics of the data
    print(df.describe())
    print('==============================================')
#     prints the data information 
    print(df.info())
    print('==============================================')
#     prints out the dtypes of the data
    print(df.dtypes)
    print('==============================================')
#     prints shape of the dataframe 
    print('Shape:')
    print(df.shape)
    print('==============================================')

    # for statment that will return the value counts for each object columns 
    for col in df.columns:
        if df[col].dtype == 'O':
            print('Value counts for column: ', col)
            print(df[col].value_counts())
            print('==========================')
        else:
            print(f'{col}: is not an object column')


def missing_data_info(df):
    missing_data = df.isnull().sum().sort_values(ascending=False)
    percent_missing = (missing_data / df.shape[0]) * 100
    missing_data_info = pd.concat([missing_data, percent_missing], axis=1, keys=['num_rows_missing', 'pct_rows_missing'])
    return missing_data_info


def handle_missing_values(df, prop_required_column, prop_required_row):
    '''
    Utilizing an input proportion for the column and rows of DataFrame df,
    drop the missing values per the axis contingent on the amount of data present.
    '''
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df