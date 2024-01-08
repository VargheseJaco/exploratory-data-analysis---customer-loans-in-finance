#%%
from pathlib import Path
import pandas as pd
from classes import *
#%%
loan_payments = pd.read_csv(Path('loan_payments.csv'))

col_names = list(loan_payments.columns)

to_date_list = ['issue_date',
                'earliest_credit_line', 
                'last_payment_date', 
                'next_payment_date', 
                'last_credit_pull_date']
to_numeric_list = []
to_categorical_list = ['term',
                       'grade', 
                       'sub_grade', 
                       'home_ownership', 
                       'verification_status', 
                       'loan_status', 
                       'purpose', 
                       'policy_code', 
                       'application_type',
                       'id',
                       'member_id']

lp_corrected_dates = DataTransform(loan_payments).to_date(to_date_list)
lp_corrected_numeric = DataTransform(lp_corrected_dates).to_numeric(to_numeric_list)
lp_corrected_dtypes = DataTransform(lp_corrected_numeric).to_categorical(to_categorical_list)


cols_to_drop = ['mths_since_last_delinq',
                'mths_since_last_record',
                'next_payment_date',
                'mths_since_last_major_derog']
lp_dropped_cols = lp_corrected_dtypes.drop(columns= cols_to_drop)

dict_to_impute = {'funded_amount': 'mean',
                  'term': 'ffill',
                  'int_rate': 'mean',
                  'employment_length': 'ffill',
                  'last_payment_date': 'median',
                  'last_credit_pull_date': 'median',
                  'collections_12_mths_ex_med': 'median'}

lp_imputed = DataFrameTransform(lp_dropped_cols).impute(dict_to_impute)
# print(DataFrameInfo(lp_imputed).nan_count())
# Plotter(lp_imputed).msno_plot()

numeric_keyslist = list(lp_imputed.skew(numeric_only=True).keys())
# Plotter(lp_imputed).hist_plot(numeric_keyslist)

exceptions_list = ['delinq_2yrs','inq_last_6mths', 'out_prncp',
               'out_prncp_inv','total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
               'collections_12_mths_ex_med']

lp_skew_fixed = DataFrameTransform(lp_imputed).skew_transform(exceptions_list)
imputed_skews = dict(lp_imputed.skew(numeric_only=True))
transformed_skews = dict(lp_skew_fixed.skew(numeric_only=True))

print(imputed_skews)
print(transformed_skews)
# Plotter(lp_skew_fixed).hist_plot(numeric_keyslist)

# Plotter(lp_skew_fixed).grid_box_plot(numeric_keyslist)

outlier_columns = ['loan_amount','funded_amount', 
                   'funded_amount_inv', 'int_rate', 
                   'instalment', 'annual_inc','dti','open_accounts',
                   'total_accounts', 'total_payment','total_payment_inv',
                   'total_rec_prncp','total_rec_int','last_payment_amount']

lp_outliers_removed = DataFrameTransform(lp_skew_fixed).outlier_removal(outlier_columns, 3)

num_removed = lp_skew_fixed.shape[0] - lp_outliers_removed.shape[0]
print('outliers removed:',num_removed)

# for i in outlier_columns:
#     Plotter(lp_skew_fixed).grid_box_plot(i)
#     Plotter(lp_outliers_removed).grid_box_plot(i)


Plotter(lp_outliers_removed).correlation_heatmap(numeric_keyslist)

lp_outliers_removed.to_csv(Path('processed_loan_payments.csv'))

# %%

