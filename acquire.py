import pandas as pd
import os
from env import get_db_url

def acquire_zillow(use_cache = True):
    '''aquire zillow dataset from the database
    '''

    filename = 'zillow.csv'

    if os.path.exists(filename) and use_cache:
        return pd.read_csv(filename, index_col=0)

    url = get_db_url('zillow')
    query = '''
            SELECT bedroomcnt AS bedrooms,
                 bathroomcnt AS bathrooms,
                 calculatedfinishedsquarefeet AS area,
                 taxvaluedollarcnt AS tax_value,
                 yearbuilt AS year_built,
                 fips
            FROM properties_2017
            JOIN propertylandusetype USING (propertylandusetypeid)
            JOIN predictions_2017 USING(parcelid)
            WHERE propertylandusedesc IN ("Single Family Residential",                       
                                            "Inferred Single Family Residential")
                AND transactiondate LIKE "2017%%";               
            '''

    df = pd.read_sql(query, url)
    df.to_csv('zillow.csv')

    return pd.read_csv('zillow.csv', index_col=0)