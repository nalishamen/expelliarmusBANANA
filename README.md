# Details about CustomDF() and RiskDataframe() from expelliarmusBANANA
This python package is BANANA!

The powerful **CustomDF()** class has 15+ methods to:

| Purpose   | Method |
|  :----  |    :----         |
|Select a subset of dataset based on a column list|select(column)|
|Return a list of column names by data type|get_columns_bydtype()|
|Assign correct data type to columns through a dictionary|SetAttributes()|
|Check columns with missing values' percentage|check_missing()|
|Check and drop duplicated rows|check_duplicated()|
|Check columns with a single value|homogenous_col()|
|Dropping rows with NaN	|drop_row_with_nas()|
|Filter columns with NaN by threshold, then drop columns| filter_threshold(threshold, more_or_less),drop_columns()
|Automatic Fill NaN based on dtype (0 /"UNKNOWN")|fill_na_bytype()|
|Manually Fill NaN with a specific value| fill_na(columns,value)|
|Drop a specific column or a list of columns |drop_columns(columns_list)|
|Encoding Binary variables|encode_binary()|

Others:
* Calcuate difference between datetimes
* Numerical calcualtion
* Extract information from a multi-labeled column
* Group: Redefine several values with a specific value
* Binary: find binary variables, encode binary variables to 0/1
* Correlation: find a list of columns that are correlated with given threshold
* Find Split based on CHAID
* Find a list of columns that are MISSING NOT AT RANDOMM with a given threshold
  
**RiskDataframe()** is specifically created to build an objective segmentation using CHAID for risk-involved Customer Segmentation.

Methods follow a workflow:

1. observation_rate(): transform into numeric values with the probability of being class 1 (target) [Applicable to Non-binary variables]
2. to_split(), find_split() Using CHAID decision tree to find statistically significant split (based on full file, thin file or a given list of input variables defined by End User) [Applicable to both numeric and non-binary variables]
3. binary_split(): transform the variable into a binary class 1=(>split point), 0=(<split point). [Applicable to both numeric and non-binary variables]
4. For each variable in full file, thin file or a given list of input variables,create Segment 1 and Segment 2 according to the binary class.
5. candidate_split_analysis():Create parameter tuned [lg_gridsearch()] Logistic Regression models for Full Population, Seg1 and Seg2,
6. find_segment_split()Create GINI Coefficients and compare the result to identify:
      Variable that is BEST Segmentation
      Variable that is Bad for segmentation but could be good for modeling
      Variable that is BAD for segmentation AND modeling 
7. Others: split_train() to split test/train
