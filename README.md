# expelliarmusBANANA
This python package is BANANA! 

The powerful CustomDF() class has 15+ methods to:
1. Load data
2. Select a subset of dataset based on a column list
3. Check duplicated rows
4. Check columns with missing values' percentage
5. Check columns with a single value homogenous_col()
NaN Treatment:
6. Drop rows with NAs for a column list
7. Fill NAs based on a column with a specific value, or by dtype (0 /"UNKNOWN")
8. Drop a specific column or a list of columns
Create new columns:
9. Calcuate difference between datetimes
10. Numerical calcualtion
11. Extract information from a multi-labeled column
12. Group: Redefine several values with a specific value
13. Binary: find binary variables, encode binary variables to 0/1
14. Correlation: find a list of columns that are correlated with given threshold
15. Find Split based on CHAID
16. Find a list of columns that are MISSING NOT AT RANDOMM with a given threshold

RiskDataframe() is specifically created to build an objective segmentation using CHAID 
for risk-involved Customer Segmentation.

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

Author:NMEN

"# expelliarmusBANANA" 
