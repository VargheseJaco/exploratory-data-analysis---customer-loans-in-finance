#%%
from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import yaml

credentials_dict = yaml.safe_load(Path('credentials.yaml').read_text())

class RDSDatabaseConnector:
    """
    A class for extracting data from the cloud.

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
        initialises engine attribute to connect to a database

    extract_to_pandas:
        reads the specified table and creates a Pandas dataframe

    data_to_csv(path):
        uses extract_to_pandas to produces a Pandas dataframe, converts to a csv and saves it to the specified path

    """
    def __init__(self, credentials):
        """
        Constructs all the necessary attributes for the connector object.

        Parameters
        ----------
            credentials : dict
                dictionary of credentials needed to connect to cloud database
        """
        self.host = credentials['RDS_HOST']
        self.password = credentials['RDS_PASSWORD']
        self.user = credentials['RDS_USER']
        self.database = credentials['RDS_DATABASE']
        self.port = credentials['RDS_PORT']

    def engine_init(self,dbtype='postgresql',dbapi='psycopg2'):
        """
        Initialises the engine attribute to connect to a database

        Parameters
        ----------
        dbtype : str
            type of database (default is postgresql)

        dbapi : str
            API used for connection

        Returns
        -------
        None
        """
        self.engine = create_engine(f"{dbtype}+{dbapi}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")

    def extract_to_pandas(self,table):
        """
        Extracts a table from the database to a Pandas dataframe

        Parameters
        ----------
        table : str
            name of the table to extract

        Returns
        -------
        a Pandas dataframe
        """
        return pd.read_sql_table(table, self.engine)
    
    def data_to_csv(self,table,path):
        """
        Extracts a table from the database to a Pandas dataframe and saves it as  a csv

        Parameters
        ----------
        table : str
            name of the table to extract

        path: Path
            filepath to write the csv to

        Returns
        -------
        None
        """
        self.extract_to_pandas(table).to_csv(path)

# %%
conn = RDSDatabaseConnector(credentials_dict)

conn.engine_init()
conn.data_to_csv('loan_payments',Path('loan_payments.csv'))

# %%
loan_payments = pd.read_csv(Path('loan_payments.csv'))
loan_payments.iloc[0]
# %%
