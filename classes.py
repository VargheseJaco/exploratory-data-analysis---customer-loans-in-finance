from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy import stats
from sqlalchemy import create_engine
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import missingno as msno
import matplotlib as mpl
import numpy as np
import pandas as pd
import psycopg2

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

class DataTransform:
    """
    A class for transforming columns in a dataframe.

    Attributes
    -----------
    dataframe: pd.DataFrame

    Methods
    -------
    to_date:
        converts column datatype to datetime
    to_numeric:
        converts column datatype to numeric
    to_categorical:
        converts column datatype to categorical
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def to_date(self,columns: list):
        """
        Converts the datatype of selected columns into datetime

        Parameters
        ----------
        columns : list
            list of columns to convert to datetime

        Returns
        -------
        a Pandas dataframe
        """
        for i in columns:
            self.dataframe[i] = pd.to_datetime(self.dataframe[i], format="mixed").dt.to_period('M')
        return self.dataframe

    def to_numeric(self,columns: list):
        """
        Converts the datatype of selected columns to numeric

        Parameters
        ----------
        columns : list
            list of columns to convert to numeric

        Returns
        -------
        a Pandas dataframe
        """
        for i in columns:
            self.dataframe[i] = pd.to_numeric(self.dataframe[1])
        return self.dataframe

    def to_categorical(self,columns: list):
        """
        Converts the datatype of selected columns to categorical

        Parameters
        ----------
        columns : list
            list of columns to convert to categorical

        Returns
        -------
        a Pandas dataframe
        """
        for i in columns:
            self.dataframe[i] = pd.Categorical(self.dataframe[i])
        return self.dataframe
      
class DataFrameInfo:
    """
    A class for extracting information about a dataframe

    Attributes
    -----------
    dataframe: pd.DataFrame

    Methods
    -------
    df_types:
        returns the datatypes of all columns in a pd.DataFrame

    mean:
        returns the mean of columns

    std_dev:
        returns the standard deviation of columns

    median:
        returns the median of columns

    unique_values:
        returns the number of unique values in columns

    df_shape:
        returns the shape of a pd.DataFrame

    nan_count:
        returns the number of null values in all columns of a pd.DataFrame
    
    nan_percentage:
        returns the percentage of null values in all columns of a pd.DataFrame
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def df_dtypes(self):
        """
        Returns the data types of a pd.DataFrame

        Parameters
        ----------
        none

        Returns
        -------
        a Pandas Series
        """
        return self.dataframe.dtypes
    
    def mean(self, columns):
        """
        Calculates the mean of columns 

        Parameters
        ----------
        columns :
            list of columns to calculate mean for
            if columns is not a list, it is converted to a list

        Returns
        -------
        a list of mean values
        """
        if type(columns) != list:
            columns = [columns]
        mean = [self.dataframe[i].mean() for i in columns]
        return mean
    
    def std_dev(self, columns):
        """
        Calculates the standard deviation of columns 

        Parameters
        ----------
        columns :
            list of columns to calculate standard deviation for
            if columns is not a list, it is converted to a list

        Returns
        -------
        a list of standard deviation values
        """
        if type(columns) != list:
            columns = [columns]
        std_dev = [self.dataframe[i].std() for i in columns]
        return std_dev

    def median(self, columns):
        """
        Calculates the median of columns 

        Parameters
        ----------
        columns :
            list of columns to calculate median for
            if columns is not a list, it is converted to a list

        Returns
        -------
        a list of median values
        """
        if type(columns) != list:
            columns = [columns]
        median = [self.dataframe[i].median() for i in columns]
        return median
    
    def unique_values(self, columns = None):
        """
        returns the number of unique values in columns

        Parameters
        ----------
        columns :
            list of columns to find number of unique values for
            defaults to None in which case the number of unique values is found for all columns
            if columns is not a list, it is converted to a list

        Returns
        -------
        a list of numbers of unique values
        """
        if columns == None:
            num_unique_values = self.dataframe.nunique()
        elif type(columns) != list:
            columns = [columns]
            num_unique_values = [self.dataframe[i].nunique() for i in columns]
        else:
            num_unique_values = [self.dataframe[i].nunique() for i in columns]
        return num_unique_values
    
    def df_shape(self):
        """
        returns shape of a dataframe

        Returns
        -------
        a tuple
        """
        print(self.dataframe.shape)
    
    def nan_count(self):
        """
        returns number of NaN values in each column of a dataframe
        
        Parameters
        ----------
        none

        Returns
        -------
        a pd.Series
        """
        return self.dataframe.isnull().sum()
    
    def nan_percentage(self):
        """
        returns percentage of NaN values in each column of a dataframe

        Parameters
        ----------
        none

        Returns
        -------
        a pd.Series
        """
        return self.dataframe.isnull().sum() * 100 / len(self.dataframe)

    def column_percentage(self, numerator, total):
        '''
        calculates the percentage of one column's sum over another column's sum.

        Parameters
        ----------
        numerator: 
            name of the column for which the percentage will be calculated
            
        total: 
            name of the column of which the total will be used as the denominator in the fraction

        Returns
        ----------
        the percentage of the
        '''
        numerator_sum = self.dataframe[numerator].sum()
        total_sum = self.dataframe[total].sum() 

        percentage = (numerator_sum / total_sum) * 100
        
        return percentage
    
    def total_recovered_over_period(self, period: int):
        
        '''
        returns a dataframe with extra information about collections

        Parameters
        ----------
        period: 
            number of months to calculate the projection for
        
        Returns
        ----------
        dataframe with added columns 'term_end', 'months_left' and 'collections_over_period'
        '''
        collections = self.dataframe.copy(deep= True)
        
        def term_end(i):
            if i['term'] == '36 months': 
                return i['issue_date'] + 36
            elif i['term'] == '60 months':
                return i['issue_date'] + 60
        
        def predicted_recovery(i):
            if i['months_left'] >= period: 
                return i['instalment'] * period 
            elif i['months_left'] < period: 
                return i['instalment'] * i['months_left']
        
        collections['term_end'] = collections.apply(term_end, axis=1)
        
        collections['months_left'] = collections['term_end'].astype(int) - collections['last_payment_date'].astype(int)
        collections = collections[collections['months_left'] > 0]
        
        collections['collections_over_period'] = collections.apply(predicted_recovery, axis=1)

        return collections      
    
    def monthly_revenue_lost(self):

        '''
        returns the cumulative revenue lost for each month of the remaining term.

        Parameters
        ----------
        none

        Returns
        ----------
        a list of the total revenue lost per month

        '''
        df_copy = self.dataframe.copy()

        def term_end(i):
            if i['term'] == '36 months': 
                return i['issue_date'] + 36
            elif i['term'] == '60 months':
                return i['issue_date'] + 60
            
        df_copy['term_end'] = df_copy.apply(term_end, axis=1)
        
        df_copy['term_left'] = df_copy['term_end'].astype(int) - df_copy['last_payment_date'].astype(int)
        
        df_copy['term_completed'] = df_copy['last_payment_date'].astype(int) - df_copy['issue_date'].astype(int)

        revenue_lost = []
        cumulative_revenue_lost = 0

        for month in range(1, (df_copy['term_left'].max()+1)):
            df_copy = df_copy[df_copy['term_left']>0]

            cumulative_revenue_lost = df_copy['instalment'].sum() 
            revenue_lost.append(cumulative_revenue_lost)

            df_copy['term_left'] = df_copy['term_left'] - 1
        
        return revenue_lost
       
class Plotter:
    """
    A class for plotting information from a dataframe

    Attributes
    -----------
    dataframe: pd.DataFrame

    Methods
    -------
    msno_plot:
        plots a graph illustrating where null values are in a dataframe

    qq_plot:
        plots QQ plots for a list of columns in a datafram
    
    hist_plot:
        plots histograms for each column in a dataframe and shows the skew value for the column
    
    hist_plot_no_skew:
        plots histograms for each column in a dataframe without showing the skew

    scatter_plot:
        plots a scatter plot between two columns
    
    grid_box_plot:
        produces a grid of box plots for all specified columns

    correlation_matrix:
        plots a correlation heatmap for th        
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def msno_plot(self):
        """
        Plots a graph illustrating where null values are in a dataframe

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        msno.matrix(self.dataframe)
    
    def qq_plot(self,columns):
        """
        Plots QQ graphs

        Parameters
        ----------
        columns:
            the specified column(s) to plot

        Returns
        -------
        none
        """
        if type(columns) != list:
            print(columns.replace("_", ' ').title(),':',self.dataframe[columns].skew(numeric_only=True))
            qq_plot = qqplot(self.dataframe[columns] , scale=1 ,line='q', fit=True)
            pyplot.show()
        else:
            for i in columns:
                print(i.replace("_", ' ').title(),':',self.dataframe[i].skew(numeric_only=True))
                qq_plot = qqplot(self.dataframe[i] , scale=1 ,line='q', fit=True)
                pyplot.show()
    
    def hist_plot(self,columns):
        """
        Plots histogram(s) and the skew of the corresponding column

        Parameters
        ----------
        columns:
            the specified column(s) to plot

        Returns
        -------
        none
        """
        if type(columns) != list:
            print(columns.replace("_", ' ').title(),':', self.dataframe[columns].skew())
            hist_plot = self.dataframe[columns].hist(bins=100)
            pyplot.show()
        else:
            for i in columns:
                print(i.replace("_", ' ').title(),':',self.dataframe[i].skew())
                hist_plot = self.dataframe[i].hist(bins=100)
                pyplot.show()

    def hist_plot_no_skew(self,columns):
        """
        Plots histogram(s) WITHOUT the skew of the corresponding column

        Parameters
        ----------
        columns:
            the specified column(s) to plot

        Returns
        -------
        none
        """
        if type(columns) != list:
            print(columns.replace("_", ' ').title())
            hist_plot = self.dataframe[columns].hist(bins=100)
            pyplot.show()
        else:
            for i in columns:
                print(i.replace("_", ' ').title())
                hist_plot = self.dataframe[i].hist(bins=100)
                pyplot.show()
                
        
    def scatter_plot(self,x,y,xlab=None,ylab=None,title=None, size = (15,13)):
        """
        Plots a scatter plot between two columns

        Parameters
        ----------
        x:
            column to use as the data for x-axis

        y:
            column to use as the data for y-axis

        Returns
        -------
        none
        """
        pyplot.figure(figsize=size)
        sns.regplot(x= x, y= y, marker= 'x',line_kws={'linewidth':0.5})
        pyplot.xlabel(xlab)
        pyplot.ylabel(ylab)
        pyplot.title(title)
        pyplot.show()

    def bar_plot(self,x,y,xlab=None,ylab=None,title=None,size=(15,13)):
        """
        Plots a bar plot between two columns

        Parameters
        ----------
        x:
            column to use as the data for x-axis

        y:
            column to use as the data for y-axis

        Returns
        -------
        none
        """
        pyplot.figure(figsize=size)
        sns.barplot(x= x, y= y)
        pyplot.xlabel(xlab)
        pyplot.ylabel(ylab)
        pyplot.title(title)
        pyplot.figure(figsize=size)
        pyplot.show()
    
    def grid_box_plot(self, column_names):
        """
        Plots a grid of box plots for the specified columns

        Parameters
        ----------
        column_names:
            columns to include in the grid

        Returns
        -------
        none
        """
        melted_df = pd.melt(self.dataframe, value_vars=column_names) 
        grid = sns.FacetGrid(melted_df, col="variable",  col_wrap=3, sharex=False, sharey=False) 
        grid = grid.map(sns.boxplot, "value", flierprops=dict(marker='+', markeredgecolor='orange')) 
        pyplot.show()
    
    def correlation_heatmap(self, columns):
        """
        Plots correlation heatmap for the columns specified

        Parameters
        ----------
        column_names:
            columns to include in the heatmap

        Returns
        -------
        none
        """
        corr = self.dataframe[columns].corr()
        pyplot.figure(figsize=(15, 13))
        sns.heatmap(corr,square=True, annot=True, fmt=".2f")
        pyplot.title('Correlation Heatmap')
        pyplot.show()

    def risk_comparison(self,column,size = (12,12)):
        '''
        plots four bar plots to illustrate the trends of a particular column in different subsets of the dataframe

        Parameters
        ----------
        column_name:
            name of the column in the dataframe to plot for
        
        Returns
        -------
           a plot consisting of four bar plots for the different subsets of the dataset
        '''
        df = self.dataframe.copy()
        paid_df = df[df['loan_status'] == 'Fully Paid']
        charged_default_df = df[df['loan_status'].isin(['Charged Off','Default'])]
        late_df = df[df['loan_status'].isin(['Late (31-120 days)','In Grace Period', 'Late (16-30 days)'])]


        overall_count = df[column].value_counts(normalize=True)
        paid_count = paid_df[column].value_counts(normalize=True)
        charged_off_default_count = charged_default_df[column].value_counts(normalize=True)
        late_count = late_df[column].value_counts(normalize=True)

        fig, axes = pyplot.subplots(nrows=4, ncols=1, figsize=size)

        axes[0].set_title('All Loans')
        axes[1].set_title('Fully Paid Loans')
        axes[2].set_title('Charged off and Default Loans')
        axes[3].set_title('Late Loans')

        sns.barplot(x=overall_count.values, y=overall_count.index, ax=axes[0]) 
        sns.barplot(x=paid_count.values, y=paid_count.index, ax=axes[1])
        sns.barplot(x=charged_off_default_count.values, y=charged_off_default_count.index, ax=axes[2])
        sns.barplot(x=late_count.values, y=late_count.index, ax=axes[3])

        pyplot.show()



class DataFrameTransform(DataFrameInfo):
    """
    A class for transforming the columns of a dataframe

    Attributes
    -----------
    dataframe: pd.DataFrame

    Methods
    -------
    impute:
        imputes null values in a column by a user-specified method

    skew_transform:
        transforms the data in a column to correct skew by the most effective method

    outlier_removal:
        removes outliers using z-score evaluation
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def impute(self, column_dict):
        """
        imputes null values in columns by a user-specified method

        Parameters
        ----------
        column_dict:
            dictionary of with column to impute as the keys and the method of imputation as the values

        Returns
        -------
        pd.Dataframe with all the columns imputed
        """
        imputed_df = self.dataframe.copy(deep= True)
        for i in column_dict:
            if column_dict[i] == 'mean':
                imputed_df[i] = imputed_df[i].fillna(self.mean(i)[0])
            elif column_dict[i] == 'median':
                imputed_df[i] = imputed_df[i].fillna(self.median(i)[0])
            elif column_dict[i] == 'ffill':
                imputed_df[i] = imputed_df[i].fillna(method= 'ffill')
            else:
                raise ValueError()
        return imputed_df
    
    def skew_transform(self, exceptions):
        """
        transforms the data in all numerical columns of a dataframe (excluding exceptions)
        to correct skew
        transform method is determined based on best performance of:
        Log Transform
        Box-Cox Transform
        Yeo-Johnson Transform

        Parameters
        ----------
        exceptions:
            list of column names that should not be transformed

        Returns
        -------
        pd.Dataframe with all the skewed columns corrected
        """
        imputed_skews = dict(self.dataframe.skew(numeric_only=True))
        to_unskew = {i:imputed_skews[i] for i in imputed_skews if imputed_skews[i]>2}
        skew_transformed = self.dataframe.copy(deep= True)
        for i in to_unskew:
            if i not in exceptions:
                log_transform = self.dataframe.copy(deep= True)[i].map(lambda i: np.log(i) if i > 0 else 0)
                yeojohnson_transform = pd.Series(stats.yeojohnson(self.dataframe.copy(deep= True)[i])[0])
                try:
                    boxcox_transform = pd.Series(stats.boxcox(self.dataframe.copy(deep= True)[i])[0])
                    skew_comparison = {'original': to_unskew[i],
                                'log': log_transform.skew(),
                                'boxcox': boxcox_transform.skew(), 
                                'yeojohnson': yeojohnson_transform.skew()}
                except ValueError:
                    skew_comparison = {'original': to_unskew[i],
                                'log': log_transform.skew(),
                                'yeojohnson': yeojohnson_transform.skew()}
                
                best_skew = min(skew_comparison.values())
                
                if best_skew == skew_comparison['log']:
                    skew_transformed[i] = log_transform
                    print(f'{i}:log:{skew_transformed[i].skew()}')
                elif best_skew == skew_comparison['yeojohnson']:
                    skew_transformed[i] = yeojohnson_transform
                    print(f'{i}:yeo:{skew_transformed[i].skew()}')
                elif best_skew == skew_comparison['boxcox']:
                    skew_transformed[i] = boxcox_transform
                    print(f'{i}:box:{skew_transformed[i].skew()}') 
                else:
                    print('original') 

        return skew_transformed
    
    def outlier_removal(self, columns, threshold):
        """
        removes rows that contain outliers in a specific column
        calculates the z-score for each datapoint in a column and removes the row if the z-score exceeds threshold
        
        Parameters
        ----------
        columns:
            columns that contain outliers to remove

        threshold:
            threshold value for the z-scores

        Returns
        -------
        pd.Dataframe with all outlier rows dropped
        """
        outliers_dropped = self.dataframe.copy(deep=True)
        if type(columns) != list:
            mean = np.mean(self.dataframe[columns]) 
            std = np.std(self.dataframe[columns]) 
            z_score = (self.dataframe[columns] - mean) / std 
            abs_z_score = pd.Series(abs(z_score)) 
            outliers_dropped = outliers_dropped[abs_z_score < threshold]
        else:
            for i in columns:   
                mean = np.mean(self.dataframe[i]) 
                std = np.std(self.dataframe[i]) 
                z_scores = (self.dataframe[i] - mean) / std 
                abs_z_scores = pd.Series(abs(z_scores))
                outliers_dropped = outliers_dropped[abs_z_scores < threshold]        
        
        return outliers_dropped
