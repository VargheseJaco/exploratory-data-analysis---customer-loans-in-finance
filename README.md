# Exploratory Data Analysis - Customer Loans in Finance

## Description:
This is an EDA project with the aim of analysing and visualising a loan payments dataset. The data is transformed to correct data types, impute null values, mitigate skew and remove outliers. The processed data is then analysed to identify key performance indicators for the loans, such as the current state of payments and projected loss.

## File Structure:
- **analysis.ipynb**: Jupyter Notebook showing the results of analysis carried out on the dataset and relevant visualisations.
- **classes.py**: Python script containing all the classes used in this project
- **db_utils.py**: Python script to extract the loan payment information from an AWS database and write it to a csv
- **loan_payments.csv**: CSV file containing the table extracted from the AWS database
- **processed_loan_payments.csv**: CSV file containing the result of all the transformations applied to loan_payments.csv
- **transformations.py**: Python script that turns loan_payments.csv into a Pandas dataframe and applies all necessary transformations. It then writes processed_loan_payments.csv

## Classes:
- **RDSDatabaseConnector**: A class for extracting data from the cloud.
- **DataTransform**: A class for transforming columns in a dataframe.
- **DataFrameInfo**: A class for extracting information about a dataframe 
- **Plotter**: A class for plotting information from a dataframe
- **DataFrameTransform**: A class for transforming the columns of a dataframe


## Data Dictionary:
- **id**: unique id of the loan
- **member_id**: id of the member to took out the loan
- **loan_amount**: amount of loan the applicant received
- **funded_amount**: The total amount committed to the loan at the point in time 
- **funded_amount_inv**: The total amount committed by investors for that loan at that point in time 
- **term**: The number of monthly payments for the loan
- **int_rate**: Interest rate on the loan
- **instalment**: The monthly payment owned by the borrower
- **grade**: LC assigned loan grade
- **sub_grade**: LC assigned loan sub grade
- **employment_length**: Employment length in years.
- **home_ownership**: The home ownership status provided by the borrower
- **annual_inc**: The annual income of the borrower
- **verification_status**: Indicates whether the borrowers income was verified by the LC or the income source was verified
- **issue_date:** Issue date of the loan
- **loan_status**: Current status of the loan
- **payment_plan**: Indicates if a payment plan is in place for the loan. Indication borrower is struggling to pay.
- **purpose**: A category provided by the borrower for the loan request.
- **dti**: A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower's self-reported monthly income.
- **delinq_2yr**: The number of 30+ days past-due payment in the borrower's credit file for the past 2 years.
- **earliest_credit_line**: The month the borrower's earliest reported credit line was opened
- **inq_last_6mths**: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
- **mths_since_last_record**: The number of months since the last public record.
- **open_accounts**: The number of open credit lines in the borrower's credit file.
- **total_accounts**: The total number of credit lines currently in the borrower's credit file
- **out_prncp**: Remaining outstanding principal for total amount funded
- **out_prncp_inv**: Remaining outstanding principal for portion of total amount funded by investors
- **total_payment**: Payments received to date for total amount funded
- **total_rec_int**: Interest received to date
- **total_rec_late_fee**: Late fees received to date
- **recoveries**: post charge off gross recovery
- **collection_recovery_fee**: post charge off collection fee
- **last_payment_date**: Last month payment was received
- **last_payment_amount**: Last total payment amount received
- **next_payment_date**: Next scheduled payment date
- **last_credit_pull_date**: The most recent month LC pulled credit for this loan
- **collections_12_mths_ex_med**: Number of collections in 12 months excluding medical collections
- **mths_since_last_major_derog**: Months since most recent 90-day or worse rating
- **policy_code**: publicly available policy_code=1 new products not publicly available policy_code=2
- **application_type**: Indicates whether the loan is an individual application or a joint application with two co-borrowers
