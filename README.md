# Fairness Linter for Python Programs
A fairness linter package written by Hongji Liu with inspirations from [UChicago CMSC 25900][1] and [IBM AI Fairness 360][2]

## Installation
**Note**: This will be a local installation of the package. 
1. Unzip the file and enter the directory where this `README.md` file is in
2. Open terminal and install the package by entering `pip3 install .` 
3. That’s it. 

## How to use it
Open any python files or environment and enter `from fairness_linter import fairness, intersectionality` which provides two functions for doing fairness testing. 

## Documentation
`fairness(data, label, pred, priv, unpriv, verbosity=1)`
The main function for doing fairness testing within a single sensitive category (e.g. race, sex, etc.)
**Parameter:**
- `data`: a pandas DataFrame that includes the necessary information of the case, and it should be dummy coded already. It should also include the actual and predicted labels as well.
- `label`: column name for the actual label; must be a string
- `pred`: column name for the predicted label; must be a string
- `priv`: column names for the privileged classes; must be a list of strings
- `unpriv`: column names for the unprivileged classes: must be a list of strings
- `verbosity`: three levels of verbosity, default=1; first level is the basic information and verdict of several tests; second level includes graphs; third level includes table of necessary data involved in calculation. 
**Return:**
Nothing. Results will be printed in the console. 

`intersectionality(data, label, pred, priv, unpriv, verbosity=1)`
The specific function for doing fairness testing that accounts for intersectionality
**Parameter:**
- `data`: a pandas DataFrame that includes the necessary information of the case, and it should be dummy coded already. It should also include the actual and predicted labels as well.
- `label`: column name for the actual label; must be a string
- `pred`: column name for the predicted label; must be a string
- `priv`: column names for the privileged classes across different categories; must be a list of strings
- `unpriv`: column names for the unprivileged classes across different categories: must be a list of strings
- `verbosity`: three levels of verbosity, default=1; first level is the basic information and verdict of several tests; second level includes graphs; third level includes table of necessary data involved in calculation. 
**Return:**
Nothing. Results will be printed in the console. 

[1]:	https://classes.cs.uchicago.edu/archive/2020/spring/25900-1/
[2]:	https://github.com/IBM/AIF360/