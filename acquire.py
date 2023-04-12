import env
import pandas as pd
import os 


def get_connection(db=env.database, user=env.username, host=env.host, pw=env.password):
    return f'mysql+pymysql://{user}:{pw}@{host}/{db}'


def get_zillow_data():
    # if the file path doesn't exist, create it
    if os.path.exists('zillow_data.csv'):
        df = pd.read_csv('zillow_data.csv',low_memory=False)
        return df
    else:
        df = pd.read_sql(env.query,get_connection())
        df.to_csv('zillow_data.csv')
        return df
        




