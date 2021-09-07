#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:30:59 2021
Updated on Tue Sep 7 17:21

@author: Nalisha_M
This document includes two class, "CustomDF" and SegmentTree
"""
import pandas as pd    
import numpy as np
import datetime
import re
#from tabulate import tabulate
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
#from CHAID import Tree, NominalColumn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


"""
    CustomDF

    Purposes:
    1. Load data
    2. Select a subset of dataset based on a column list
    3. Check duplicated rows
    4. Check columns with missing values' percentage
    5. Drop rows with NAs for a column list
    6. Fill NAs based on a column with a specific value
    7. Drop a specific column or a list of columns

"""



class CustomDF():

    def set_df(self, data):
        self.data = data
    def get_df(self):
        return self
    
    def select(self, column):
        """
        Select a subset of dataset based on the list pass through 'column'
        """
        if isinstance(column, list):
            return self.data[column]
        
    def get_columns_bydtype(self, data_types, to_ignore = list(), ignore_target = False):
        columns = self.data.select_dtypes(include=data_types).columns
        if ignore_target:
            columns = filter(lambda x: x not in to_ignore, list(columns))
        return list(columns)
    
    def SetAttributes(self,argument_dict):
        for key,value in argument_dict.items():
            self.data[key]=self.data[key].astype(value)
        
#-----------------------------------------------------------------------------
                        # DATA HANDLING
#-----------------------------------------------------------------------------
   
    def check_duplicated(self):
        """
        Detect for duplicated rows.
        """
        if self.data.duplicated(keep=False).sum()==0:
            print("There is no ENTIRELY duplicated rows.")
        else:
            print (str(self.data.duplicated(keep=False).sum())+" duplicated rows are dropped.")
            self.data.drop_duplicates()
            return self

    def check_missing(self):
        """
        To check the column's missing value and its percentage
        """
        miss_pres=round(self.data.isnull().sum()/self.data.shape[0]*100,2).sort_values(ascending=False)>0
        miss_pres_val=round(self.data.isnull().sum()/self.data.shape[0]*100,2).sort_values(ascending=False)[miss_pres]
        if len(miss_pres_val) ==0:
            print('There is no missing variable.')
        else:
            print(miss_pres_val)
            
    def filter_threshold(self,threshold, more_or_less):
        """
        Filter the columns based on missing values' percentage, bigger or smaller, than the given threshold.
        """
        miss_pres=round(self.data.isnull().sum()/self.data.shape[0]*100,2).sort_values(ascending=False)>0
        miss_pres_val=round(self.data.isnull().sum()/self.data.shape[0]*100,2).sort_values(ascending=False)[miss_pres]
        if more_or_less.lower()=="more":
            return list(miss_pres_val[miss_pres_val>threshold].index)
        elif more_or_less.lower()=="less":
            return list(miss_pres_val[miss_pres_val<threshold].index)
        else:
            print("Error: check the function arguments again.")        
   
    def drop_row_with_nas(self,column):
        """
        Drop rows with NA by given column or column list.
        """
        if isinstance(column, list) is not True:
            column = [column]
        for col in column:
            list((self.data[col].isnull()).index)
            self.data.drop(labels=list(self.data.index[self.data[col].isnull()]),axis=0, inplace=True)
        self.data = self.data.reset_index(drop=True)
        return self
    
    def homogenous_col(self):
        homogenous_columns=[]
        for col in self.data.columns:
            if len(self.data[col].value_counts()) == 1:
                homogenous_columns.append(col)
        return homogenous_columns
      
    def fill_na(self, column, value):
        """
        Fill missing value under the column with a specific value 
        """
        if isinstance(column, list) is True:
            for col in column:
                self.data[col].fillna(value, inplace=True)
        else:
            self.data[column].fillna(value, inplace=True)
        return self
    
    def fill_na_bytype(self):
        """
        Fill NaN by numeric (0) or categorical columns ("UNKNOWN").
        
        """
        miss_ls=list(self.data.columns[self.data.isna().sum()!=0])
        miss_type=self.data[miss_ls].dtypes.to_list()
        #Check the datatype for those missing columns, if the column is numeric, fill 0, if the column is categorical, fill "A"-to be LabelEcoded into 0 later.
        self.fill_na(list(self.data[miss_ls].select_dtypes(include=[np.number]).columns),0)
        self.fill_na(list(self.data[miss_ls].select_dtypes(exclude=[np.number]).columns),'UNKNOWN')

    def drop_columns(self, columns_list):
        """
        Drop a column or a list of columns indicated by "columns_list" argument.
        """
        if isinstance(columns_list, list) is not True:
            columns_list = [columns_list]
        for column in columns_list:
            if column in columns_list:
                self.data.drop(column, axis=1, inplace=True)
        return self
    
        """
        Functions for Data Cleaning and transformation 

        """
    def redefine_group(self, column, old_values, new_value):
        """
        Re-group values within a categorical column, e.g Profession or Car Type
        """
        self.data[column] = self.data[column].apply(
            lambda x: new_value if x in old_values else x).astype('object')
        return self
    
    def date_difference(self,begin,end,form,new_name):
        """
        Given a column of beginning date, and string value form (default as "day")
        to indicate "year", "month", and also a new column name
        Return a new column with differences.
        
        E.g Age from BIRTHDAY and today
        
        """
        if form.lower() =="year":
            d=365
        elif form.lower()=='month':
            d=30
        else:
            d=1
        if end.lower()!= "today":
            self.data[new_name]=round((self.data[end]-self.data[begin]).dt.days/d)
        else:
            self.data[new_name]=round((pd.datetime.today() - self.data[begin]).dt.days/d)
        return self

    def calculation(self, a, b, new_name, method):
        """
        Create a new column with calculation result from identified method.
        method:
            division /
            add +
            substruct -
            multiple *
        """
        if method == "/":
            self.data[new_name]=self.data[a]/self.data[b]
        elif method=="-":
            self.data[new_name]=self.data[a]-self.data[b]
        elif method=="x" or method=="*":
            self.data[new_name]=self.data[a] * self.data[b]
        else:
            self.data[new_name]=self.data[a] + self.data[b]
        return self
    
    def existing_cr(self,customer_open,loan_open):
        """
        Based on customer open date and Loan open date to determine whether
        there has been an existing customer relationship

        """
        self.data["EXISTING_CR"]= np.where((self.data[customer_open] > self.data[loan_open]), "Yes", "No")
        return self
    
    def extractinfo(self,column,re_expression,new_name):
        """
        Extract information from regular expression or a certain word from a column, 
        Create a new column based on it.
        """
        self.data[new_name]=self.data[column].str.extract(pat = re_expression)
        return self
    
    def to_dummies(self,column):
        """
        Turn a given column into dummies variables
        """
        df1=pd.get_dummies(self.data[column])
        for col in list(df1.columns):
            self.data[col]=df1[col]
        self.data.drop(column,axis=1, inplace=True)
        return self
    
    
    def find_split(self,independent_variable_columns,dep_variable,target):
        from CHAID import Tree, NominalColumn
        tree = Tree.from_pandas_df(self.data, dict(zip(independent_variable_columns, 
                        ['nominal'] * len(independent_variable_columns))), dep_variable,min_parent_node_size=2,
                        dep_variable_type='continuous', max_depth=5)
        #tree.print_tree()
        ## to get a LibTree object,
        tree.to_tree()
        num=0
        for a in range(0,len(tree.tree_store)):
            if tree.tree_store[a].split.column==target:
                split=tree.tree_store[a].members['mean']
            else:
                num=num+1
        if num==len(tree.tree_store):
            print("Not good for segmentation after Chi-square Analysis.")
        else:
            print("There is a split for "+ str(dep_variable)+" variable at: "+str(split))
            return split
    
    
    def find_corr(self,threshold=0.75):
        import seaborn as sns 
        '''
        To find correlated numeric variables
        Return a heatmap and printed message if any.
        '''
        numeric_variables=self.get_columns_bydtype(["float64", "int64"], ignore_target = False)
        cormat = self.data[numeric_variables].corr()
        round(cormat,2)
        corva=cormat[cormat.abs()>threshold].sum()!=1
        if sum(corva)>0:
            print("At threshold of "+str(threshold)+",these variables are identified: "+ str(list(corva[corva==True].keys())))
            sns.heatmap(cormat)
            return list(corva[corva==True].keys())
        else:
            print("There is no correlated numeric variables.")
        
     
      
    def find_binary(self):
        """
        To find binary categorical variables and return a list of them.
        """
        categorical_variables=self.get_columns_bydtype(["object"])
        #Select binary categorical variables to be labelcoded as 1 or 0
        binary=[]
        for col in categorical_variables:
            if len(self.data[col].value_counts())==2:
                binary.append(col)
        if len(binary)!=0:
            print("The binary variables are: " + str(binary))
        return binary
    
    def encode_binary(self):
        """
        Encode Binary variables to 1/0
        """
        from sklearn.preprocessing import LabelEncoder
        binary=self.find_binary()
        categorical_variables=self.get_columns_bydtype(["object"])
        non_binary=list(set(categorical_variables)-set(binary))
        
        if len(binary)!=0:
            for col in binary:
                self.data[col] = LabelEncoder().fit_transform(self.data[col])
            if len(non_binary)==0:
                print("All " + str(len(binary))+ " categorical variables are encoded.")
        else:
            for col in non_binary:
                self.data[col] = LabelEncoder().fit_transform(self.data[col])
            print("Non_binary ones: "+ str(non_binary)+" are encoded.")

#-----------------------------------------------------------------------------
                        # RISK BASED APPROACH
#----------------------------------------------------------------------------- 
    def missing_not_at_random(self,input_vars,target,threshold):
        if len(input_vars)==0:
            input_vars=list(self.data.columns)
            
        for col in input_vars:
            self.data[col]=self.data[col].fillna(value='Missing')
    
        dfmiss=pd.DataFrame(self.data)
        missing=list(dfmiss.columns)
        for col in missing:
            dfmiss[col]=dfmiss[col].apply(lambda x: 0 if x!='Missing' else 1)

        cormat = dfmiss[input_vars].corr()
        round(cormat,2)
        corva=cormat[cormat.abs()>threshold].sum()!=1
        if len(corva)!=0:
            variables= list(corva[corva==True].keys())
        else:
            variables=[]
            
        full_file=self.data.columns.tolist()
        full_file.remove(target)
        thin_file=list(set(full_file)-set(variables))
    
        if len(variables)>0:
            print("--"*25)
            print("\033[1m"+" Missing Not At Random Report  "+'\033[0m')
            print("--"*25)
            print("\033[1m"+"> Missing Not At Random Features are: "+'\033[0m'+ ','.join(sorted(variables))+" at threshold "+ str(threshold))
            print('\n')
            print("\033[1m"+">> Target is "+'\033[0m'+ str(target)+". Therefore we recommend:")
            print('\n')
            print("\033[1m"+">> Thin File Segmentation Variables are: "+'\033[0m',', '.join(sorted(thin_file)))
            print('\n')
            print("\033[1m"+">> Full File Segmentation Variables are: "+'\033[0m',', '.join(sorted(full_file)))
            
        else:
            thin_file=[]
            print("There is no MNAR variables identified. A full file is suggested.")
            print("\033[1m"+">> Full File Segmentation Variables are: "+'\033[0m',', '.join(sorted(full_file)))
        return full_file,thin_file
