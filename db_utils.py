#%%
from pathlib import Path
import yaml
from classes import RDSDatabaseConnector

credentials_dict = yaml.safe_load(Path('credentials.yaml').read_text())

conn = RDSDatabaseConnector(credentials_dict)

conn.engine_init()
conn.data_to_csv('loan_payments',Path('loan_payments.csv'))
