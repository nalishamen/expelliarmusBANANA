#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:30:59 2021
Updated on Tue Sep 11 17:21 2021

@author: Nalisha_M
This document includes two class, "CustomDF" and "RiskDataframe"
"""
import pandas as pd    
import numpy as np
import datetime
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from tabulate import tabulate
from CHAID import Tree, NominalColumn

"""
    CustomDF

    Purposes:
    1. Load data
    2. Select a subset of dataset based on a column list
    3. Check duplicated rows
    4. Check columns with missing values' percentage
    5. Check columns with a single value homogenous_col()
    6. NaN Treatment:
        Drop rows with NAs for a column list
        Fill NAs based on a column with a specific value, or by dtype (0 /"UNKNOWN")
        Drop a specific column or a list of columns
    7. Create new columns:
        Calcuate difference between datetimes
        Numerical calcualtion
        Extract information from a multi-labeled column
    8. Group: Redefine several values with a specific value
    9. Binary: find binary variables, encode binary variables to 0/1
    10. Correlation: find a list of columns that are correlated with given threshold
    11. Find Split based on CHAID
    12. Find a list of columns that are MISSING NOT AT RANDOMM with a given threshold

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
        binary=[]
        for col in self.data.columns:
            if len(self.data[col].value_counts())==2:
                binary.append(col)
        return binary
    
    def encode_binary(self,var=[]):
        """
        Encode Binary variables to 1/0
        """
        from sklearn.preprocessing import LabelEncoder
        binary=self.find_binary()
        if len(var)!=0:
            for a in var:
                if len(self.data[a].value_counts())==2:
                    self.data[a]=LabelEncoder().fit_transform(self.data[a])
                    self.data[a]=self.data[a].astype('category')
                    print("Binary encoding is done for: " + str(var)+ ".")
                else:
                    raise ValueError("Input variable is not binary, please check it again!")
                
        elif len(binary)!=0:
            for col in binary:
                self.data[col] = LabelEncoder().fit_transform(self.data[col])
                self.data[col]=self.data[col].astype('category')
                print("Binary encoding is done for: " + str(var)+ ".")
        else:
            print("No variable is binary.")

#-----------------------------------------------------------------------------
                        # MISSING NOT AT RANDOM (MNAR)
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



class RiskDataframe(pd.DataFrame):
    """
    Customer segmentation involves categorizing the portfolio by industry, location, revenue, account size, and number of employees and many other variables
    to reveal where risk and opportunity live within the portfolio.

    The class is created specifically for Customer Segmentation by building an objective segmentation using CHAID.

    A CustomDF() that has transfromed all datetime columns into numeric ones and transformed all numeric columns that are correlated.

    Target is selected and is encoded into binary 1/0.

    NaN are treated after performing Missing Not At Random in CustomDF()

    full_file and thin_file are created by MNAR Analysis.
    
    Methods follow a workflow:
    1. observation_rate(): transform into numeric values with the probability of being class 1 (target)
	Applicable to Non-binary variables
    2. to_split(), find_split() Using CHAID decision tree to find statistically significant split (based on full file, thin file or a given list of input variables defined by End User)
	Applicable to both numeric and non-binary variables
    3. transform the variable into a binary class 1=(>split point), 0=(<split point).
	Applicable to both numeric and non-binary variables
    4. binary_split(): For each variable in full file, thin file or a given list of input variables,create Segment 1 and Segment 2 according to the binary class.
    5. candidate_split_analysis():Create parameter tuned [lg_gridsearch()] Logistic Regression models for Full Population, Seg1 and Seg2, 
    6. find_segment_split()Create GINI Coefficients and compare the result to identify:
	- Variable that is BEST Segmentation
	- Variable that is Bad for segmentation but could be good for modeling
	- Variable that is BAD for segmentation AND modeling
    Others: split_train() to split test/train
        

    
    """
    

    #Initializing the inherited pd.DataFrame
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
    
    @property
    def _constructor(self):
        def func_(*args,**kwargs):
            df = RiskDataframe(*args,**kwargs)
            return df
        return func_

    def set_df(self, df, target,full_file,thin_file):
        self.df=df
        self.data=df.data
        self.target=target
        self.file=self.data.columns.to_list()
        self.numeric_variables = self.df.get_columns_bydtype(["float64", "int64"], [target], ignore_target = True)
        self.categorical_variables= list(set(list(set(self.file)-set(self.numeric_variables)))-set([self.target]))
        self.binary=self.find_binary()
        self.file_t=list(set(self.file)-set([self.target]))
        self.nominal= list(set(self.categorical_variables)-set(self.binary))
        self.thin=thin_file
        self.full=full_file
        self.indep_ls= self.binary+[self.target]


#-----------------------------------------------------------------------------
                        # RISK BASED APPROACH
#-----------------------------------------------------------------------------    
    
    def split_train(self,size=0.5):
        '''
        Purpose: To split the dataset into train and test sets, with given test_size.
        Parameters: size is test_size with default value set to be 0.5
        Returns: df_train, df_test
        '''
        splitter = train_test_split
        df_train, df_test = splitter(self.data, test_size = size, random_state = 42)
        return df_train,df_test

    def find_segment_split(self, input_vars=[],candidate=[],ordered_nominal=[]):
        """

        Returns
        -------
        Example 1: Not good segmentation found for: SEX,but should be a good feature in modeling.
        >>Segment 1: SEX in (M) [GINI Full Model: 32.1627% / GINI Segmented Model: 31.6210%]
        >>Segment 2: SEX in (F) [GINI Full Model: 30.8930% / GINI Segmented Model: 30.8321%]
                
    
        candidate: if a list of variables is not given, default setting to go through all full file variables.
        input_vars: [] if input_vars is not defined specifically, full file variables are chosen;
                "thin" -- thin file variables are chosen
        ordered_nominal: if any non-binary categorical is ordered, specify the column names into a list.
        """
        if len(candidate)==0:
            candidate=self.full
        if len(input_vars)==0:
            all_variables=self.full
        else:
            all_variables=[]
            for a in input_vars:
                if a in self.file:
                    all_variables.append(a)
                else:
                    raise ValueError("Wrong input variable, please check it again! Select variables from:"+str(self.full))
        split=self.dataset_split(all_variables,ordered_nominal)
        print("\033[1m"+"Segmentation Analysis: Variable by Variable Check for "+'\033[0m'+str(all_variables))
        for can in candidate:
            criteria=pd.DataFrame({"seg1":self.data[can]==1,"seg2":self.data[can]==0})
            self.candidate_split_analysis(can,criteria,split[can],all_variables)
        
    
    def candidate_split_analysis(self,candidate,criteria,split,all_variables):
        if self.data[candidate].dtype=='datetime64[ns]' or self.data[candidate].dtype=='str':
            raise DataTypeError ("The candidate variable has WRONG data type. Dates or strings have to be transformed before using this method.")
    
        if np.var(self.data[candidate])<=0.1:
            print("--"*50)
            print("There is no statistically significant split found for candidate "+"\033[1m"+str(candidate)+'\033[0m'+ ", with variance (<0.1).")
            print("--"*50)
        else:

            

            '''
            Step 1: Split the train/test sets for df through method split_train()

            '''
            df_train, df_test = self.split_train(0.4)
            X_train = df_train[all_variables]
            y_train = df_train[self.target]
            X_test = df_test[all_variables]
            y_test = df_test[self.target]

            
            para_full=lg_gridsearch(X_train,y_train)
            method_full=LogisticRegression(dual=para_full['dual'],
                                            C=para_full['C'],
                                            max_iter=para_full['max_iter'])
            fitted_full_model=method_full.fit(X_train, y_train)
        
            y_pred = fitted_full_model.predict_proba(X_test)


            X_train = df_train[all_variables]
            y_train = df_train[self.target]
            X_test = df_test[all_variables]
            y_test = df_test[self.target]

            df_train_seg1 = df_train[criteria["seg1"]]
            df_train_seg2 = df_train[criteria["seg2"]]
            df_test_seg1 = df_test[criteria["seg1"]]
            df_test_seg2 = df_test[criteria["seg2"]]
    
            X_train_seg1 = df_train_seg1[all_variables]
            y_train_seg1 = df_train_seg1[self.target]
            X_test_seg1 = df_test_seg1[all_variables]
            y_test_seg1 = df_test_seg1[self.target]

            X_train_seg2 = df_train_seg2[all_variables]
            y_train_seg2 = df_train_seg2[self.target]
            X_test_seg2 = df_test_seg2[all_variables]
            y_test_seg2 = df_test_seg2[self.target]
        
            para_seg1=lg_gridsearch(X_train_seg1,y_train_seg1)
            method_seg1=LogisticRegression(dual=para_seg1['dual'],
                                       C=para_seg1['C'],
                                      max_iter=para_seg1['max_iter'])
        
            para_seg2=lg_gridsearch(X_train_seg2,y_train_seg2)
            method_seg2=LogisticRegression(dual=para_seg2['dual'],
                                       C=para_seg2['C'],
                                      max_iter=para_seg2['max_iter'])
            fitted_model_seg1 = method_seg1.fit(X_train_seg1, y_train_seg1)
            fitted_model_seg2 = method_seg2.fit(X_train_seg2, y_train_seg2)
        
            y_pred_seg2_proba = fitted_model_seg2.predict_proba(X_test_seg2)[:,1]
            y_pred_seg2_fullmodel_proba = fitted_full_model.predict_proba(X_test_seg2)[:,1]

            y_pred_seg1_proba = fitted_model_seg1.predict_proba(X_test_seg1)[:,1]
            y_pred_seg1_fullmodel_proba = fitted_full_model.predict_proba(X_test_seg1)[:,1]

            G1=gini(y_test_seg1, y_pred_seg1_proba)*100
            G2=gini(y_test_seg1, y_pred_seg1_fullmodel_proba)*100
            G3=gini(y_test_seg2, y_pred_seg2_proba)*100
            G4=gini(y_test_seg2, y_pred_seg2_fullmodel_proba)*100
            G0 =gini(y_test,y_pred[:,1])*100
    
            if type(split)==list or type(split)==tuple:
                sg1=split[0]
                sg2=split[1]
            else:
                sg1=" > "+str(split)
                sg2=" < "+str(split)
        

            if G2>G1 and G4>G3:
                print("--"*50)
                print("\033[1m"+"BEST segmentation found for: "+str(candidate)+'\033[0m')
                print('\n')
                print("\033[1m"+">>Segment 1: "+str(candidate)+'\033[0m'+ " in ("+str(sg1)+ ")"+"\033[1m"+" [GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(G1,G2)+'\033[0m')
                print('\n')
                print("\033[1m"+">>Segment 2: "+str(candidate)+'\033[0m'+ " in ("+str(sg2)+")"+"\033[1m"+" [GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(G3,G4)+'\033[0m')
                print('\n')
            elif G2>G0 or G4>G0:
                print("--"*50)
                print("\033[1m"+"Not good segmentation found for: "+str(candidate)+",but should be a good feature in modeling."+'\033[0m')
                print('\n')
                print("\033[1m"+">>Segment 1: "+str(candidate)+'\033[0m'+ " in ("+str(sg1)+")"+"\033[1m"+" [GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(G1,G2)+'\033[0m')
                print("\033[1m"+">>Segment 2: "+str(candidate)+'\033[0m'+" in ("+str(sg2)+")"+"\033[1m"+" [GINI Full Model: {:.4f}% / GINI Segmented Model: {:.4f}%]".format(G3,G4)+'\033[0m')
                print('\n')
            else:
                print("--"*50)
                print("After analysis, "+str(candidate)+ " segmented by ["+ "\033[1m"+str(sg1)+"] / ["+str(sg2)+'\033[0m'+"] is not good for segmentation nor ideal for modeling.")
                print('\n')
    
    def find_binary(self):
        """
        To find binary categorical variables and return a list of them.
        """
        binary=[]
        for col in self.categorical_variables:
            if len(self.data[col].value_counts())==2:
                binary.append(col)
        return binary

    def observation_rate(self,col):
        items=list(self.data[col].value_counts().keys())
        rate={}
        if len(items)>2:
            totalnum=len(self.data[col])
            for a in range(len(items)):
                rate.update({items[a]:self.data[self.data[col]==items[a]][self.target].astype('int').sum()/totalnum})
                for key,value in rate.items():
                    self.data[col]=self.data[col].replace(key,value)
            #print(col,rate)
            return rate
        elif len(items)==2:
            print(str(col)+" is a binary variable, no need to the transformation.")

    def binary_split(self,column,split_pt,rate={}):
        
        self.data[column]=self.data[column].apply(lambda x:1 if x>split_pt else 0)
        if len(rate)!=0:
            seg1_cat=[]
            seg2_cat=[]
            for key,value in rate.items():
                if value>split_pt:
                    seg1_cat.append(key)
                if value<=split_pt:
                    seg2_cat.append(key)
            #print("First split at "+str(seg1_cat))
            #print("Second split at "+str(seg2_cat))
            return [seg1_cat,seg2_cat]
    
    def find_split(self,dep_variable,input_var):
        """
        Applying CHAID decision tree to find valid split 
        for continuous numerical variables and nonbinary variables that have been transformed into probability
        of class 1.

        
        """
        split=0
        if input_var!=self.full:
            indep_ls=[]
            for a in input_var:
                if a in self.indep_ls:
                    indep_ls.append(a)
        else:
            indep_ls=self.indep_ls
            
        tree = Tree.from_pandas_df(self.data, dict(zip(indep_ls,
                        ['nominal'] * len(indep_ls))), dep_variable,min_parent_node_size=2,
                        dep_variable_type='continuous', max_depth=6)
        tree.to_tree()
        for a in range(0,len(tree.tree_store)):
            if tree.tree_store[a].split.column==self.target:
                split=tree.tree_store[a].members['mean']
        #if split==0:
            #print("No statistically significant split is found for "+str(dep_variable))
        return split
        
        
    def to_split(self,input_var=[],order_nominal=[]):
        """
        A method to split a Segment():
            numerical variables: binary_split(), then return split point
            non-binary categorical variables: find the split, binary_split(),then return seg1,seg2
    
        """
        split={}
        if len(order_nominal)!=0:
            for var in order_nominal:
                rate={}
                le = preprocessing.LabelEncoder()
                le.fit(self.data[var])
                le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                rate.update({var:le_name_mapping})
                self.data[var]=le.fit_transform(self.data[var])
                order=self.find_split(var,input_var)
                seg_list=self.binary_split(var,order,rate[var])
                split.update({var:seg_list})
            for var in self.numeric_variables:
                order=self.find_split(var,input_var)
                self.binary_split(var,order)
                split.update({var:order})
            
            for var in self.nominal:
                if var not in order_nominal:
                    rate=self.observation_rate(var)
                    order=self.find_split(var,input_var)
                    if order == None:
                        order=self.data[var].mean()
                    seg_list=self.binary_split(var,order,rate)
                    split.update({var:seg_list})
        else:
            for var in self.numeric_variables:
                order=self.find_split(var,input_var)
                self.binary_split(var,order)
                split.update({var:order})
            
            for var in self.nominal:
                rate=self.observation_rate(var)
                order=self.find_split(var,input_var)
                if order == None:
                    order=self.data[var].mean()
                seg_list=self.binary_split(var,order,rate)
                split.update({var:seg_list})
           
        if len(split)!=0:
            return split
        else:
            print("Not good for segmentation after Chi-square Analysis on "+str(list(set(set(self.file)-set(self.binary))-set([self.target]))))
    

        
    def dataset_split(self,input_vars=[],ordered_nominal=[]):
        split={}
    
        binary=self.find_binary()
        if self.target in binary:
            binary.remove(self.target)
        for col in binary:
            split.update({col:tuple(self.data[col].value_counts().keys())})
            self.data[col]=LabelEncoder().fit_transform(self.data[col])
        split_stat=self.to_split(input_vars,ordered_nominal)
        for key,value in split_stat.items():
            split.update({key:value})
        return split
        

def gini(a,b):
    
    """
    >2 parameters:
        a is a binary variable
        b is the prediction of a.
    < the gini of the two.
    e.g gini=2*roc_auc_score(y_test, pred_prob1[:,1])-1
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(a, b)
    roc_auc = auc(fpr, tpr) 
    GINI = (2 * roc_auc) - 1
    return GINI

def lg_gridsearch(X_train,y_train):
        dual=[True,False]
        max_iter=[100,110,120,130,140,150]
        C = [1.0,1.5,2.0,2.5]
        param_grid = dict(dual=dual,max_iter=max_iter,C=C)

        lr = LogisticRegression(penalty='l2')
        grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)
        return grid_result.best_params_
    

