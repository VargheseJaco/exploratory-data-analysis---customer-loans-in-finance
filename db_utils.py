#%%
import pandas as pd
from pathlib import Path
import psycopg2
from sqlalchemy import create_engine
import yaml

credentials_dict = yaml.safe_load(Path('credentials.yaml').read_text())

class RDSDatabaseConnector:
    """
    A class to extract the data from the cloud.

    Attributes
    -----------
    host: str
        host endpoint
    password: str
        password for database
    user: str
        username for database login
    database: str
        name of database
    port: int
        port number for endpoint

    Methods
    -------
    engine_init(dbtype='postgresql',dbapi='psycopg2'):
        initialises engine attribute to connect to a database with default strings for the type of database(dbtype) and API(dbapi)

    extract_to_pandas:
        reads the specified table and creates a Pandas dataframe

    data_to_csv(path):
        uses extract_to_pandas to produces a Pandas dataframe, converts to a csv and saves it to the specified path

    """
    def __init__(self, credentials):
        self.host = credentials['RDS_HOST']
        self.password = credentials['RDS_PASSWORD']
        self.user = credentials['RDS_USER']
        self.database = credentials['RDS_DATABASE']
        self.port = credentials['RDS_PORT']

    def engine_init(self,dbtype='postgresql',dbapi='psycopg2'):
        self.engine = create_engine(f"{dbtype}+{dbapi}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")

    def extract_to_pandas(self):
        return pd.read_sql_table('loan_payments', self.engine)
    
    def data_to_csv(self,path):
        self.extract_to_pandas().to_csv(path)

# %%
conn = RDSDatabaseConnector(credentials_dict)

conn.engine_init()
conn.data_to_csv(Path('loan_payments.csv'))

# %%
loan_payments = pd.read_csv(Path('loan_payments.csv'))
loan_payments.head(5)
# %%
