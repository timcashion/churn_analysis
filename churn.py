# basics
import numpy as np
import pandas as pd
from itertools import chain


# classifiers / models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# other
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss, classification_report, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold


dat = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
dat.head()
dat.describe()
dat = dat.drop(columns=["customerID"])

#Clean datasets and encode where necessary:
dat.Churn = dat.Churn.map(dict(Yes=1, No=0))

#Filter out " " Total Charges:
len(dat)
#dat = dat[pd.isna(dat.TotalCharges)==False] #I thought they were NAs but they are " "
dat = dat[dat.TotalCharges != " "]
len(dat)

dat.dtypes #Check data types for cleaning and encoding
numeric_cols = ["MonthlyCharges", "TotalCharges"]
integer_cols = ["SeniorCitizen", "tenure"]
categorical_cols = ["gender", 'Partner', "Dependents", "PhoneService", 
"MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
"StreamingTV", "StreamingMovies", "Contract",
 "PaperlessBilling", "PaymentMethod"]

dat = pd.get_dummies(dat, columns=categorical_cols, drop_first=True)
dat[numeric_cols] = dat[numeric_cols].astype(float)
dat[integer_cols] = dat[integer_cols].astype(int)

#Do we need to normalize our numeric and integer cols?
#SeniorCitizen was already in binary form. So no. 

from plotnine import ggplot, aes, geom_histogram
(ggplot(dat, aes(x='MonthlyCharges'))
+ geom_histogram())

(ggplot(dat, aes(x='TotalCharges'))
+ geom_histogram())

#Neither follow a normal distribution. Log transformation could help, but these are odd. 
dat["LogTotalCharges"] = np.log(dat["TotalCharges"]+1)
dat["LogMonthlyCharges"] = np.log(dat["MonthlyCharges"]+1)


(ggplot(dat, aes(x='LogMonthlyCharges'))
+ geom_histogram())

(ggplot(dat, aes(x='LogTotalCharges'))
+ geom_histogram())

#Doesn't really help so leave this for now. 

dat = dat.drop(columns = ["LogTotalCharges", "LogMonthlyCharges"])

#Split to X and y:
y = dat.Churn
X = dat.drop(columns=["Churn"]) #Drop Dependent and ID column 

#Split data into 3 subsets:
# Split first 80/20 then 75/25 to get a 60/20/20 data split.
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, stratify=y_train_valid, train_size=0.75, random_state=42)

#Test splits capture all data points:
len(dat) == len(X_train) + len(X_valid) + len(X_test)
#Test splits are roughly stratified based on y:
ys = [y, y_valid, y_train, y_test]
[round(np.mean(y), ndigits=4) for y in ys]


#Fit and Report function:
def fit_and_report_errors(model, X, y, Xv, yv, mode = 'classification'):
    """
    Original source: 
    https://github.ubc.ca/MDS-2019-20/DSCI_573_feat-model-select_instructors/tree/master/source
    
    fits a model and returns train and validation errors
    
    ---------     
    model -- sklearn classifier model
        The sklearn model
    X -- numpy.ndarray        
        The X part of the train set
    y -- numpy.ndarray
        The y part of the train set
    Xv -- numpy.ndarray        
        The X part of the validation set
    yv -- numpy.ndarray
        The y part of the validation set       
    
    Keyword arguments 
    -----------------
    mode -- str 
        The mode for calculating error (default = 'regression') 
        TC: Changed default mode to classification for this analysis

    Returns
    -------
    errors -- list
        A list containing train (on X, y) and validation (on Xv, yv) errors
    
    """
    model.fit(X, y)
    if mode.lower().startswith('regress'):
        errors = [mean_squared_error(y, model.predict(X)), mean_squared_error(yv, model.predict(Xv))]
    if mode.lower().startswith('classif'):
        errors = [1 - model.score(X, y), 1 - model.score(Xv, yv)]    
    return errors



#Establish baseline/dummy model:
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier()
dummy_errors = fit_and_report_errors(dummy, X_train, y_train, X_valid, y_valid)

#Set up dictionary for storing various models and their errors:
model_dict = {}
model_dict["dummy"] = dummy_errors

#Test basic models without hyper parameter optimization yet: 
models = {'Logistic regression': LogisticRegression(),
    'RBF SVM' : SVC(), 
    'random forest' : RandomForestClassifier() , 
    'neural net' : MLPClassifier()
}

#For loop to go over basic models
for model_name, model in models.items():
    errors = fit_and_report_errors(model, X_train, y_train, X_valid, y_valid)
    model_dict[model_name] = errors #Save errors in model dictionary

#Convert model scores to dataframe for easy viewing of results 
model_results = pd.DataFrame(model_dict).T
model_results.columns = ["train_error","test_error"]
model_results.train_error = round(model_results.train_error, ndigits=4)
model_results.test_error = round(model_results.test_error, ndigits=4)
model_results.to_csv("model_results.csv")

model_results.columns
#All models show an improvement over dummy model, with some (i.e., RBF SVM and Random Forest) showing HIGH signs of overfitting. 
#Next steps would be to re-run with some hyper parameter testing to see where I can find improvements 
#Also move beyond simple metric of model error to full confusion matrix to make sure I'm making progress in the right areas.    

#What it means? 
#Simple way to see what's important is what features (that we have in the dataset) can help us understand customer churn beheaviour
#A simple way to do this is with Logistic Regression with the coefficients being an indication of feature importance
lr = LogisticRegression()
lr.fit(X_train, y_train)

np.round(abs(lr.coef_), decimals=2)
important_features = (np.round(abs(lr.coef_), decimals=2) > 0.2).tolist()

coefs = lr.coef_.tolist()[0]

zipped_cols_coefs = zip(X_train.columns, coefs)
coef_plot_data = pd.DataFrame(zipped_cols_coefs)
coef_plot_data.columns = ["Variable", "Coefficient"]
coef_plot_data = coef_plot_data.sort_values('Coefficient')
#coef_plot_data["Variable_name"] = coef_plot_data["Variable"]

var_ordered = coef_plot_data['Variable'][coef_plot_data['Coefficient'].sort_values().index.tolist()] 
coef_plot_data['Variable'] = pd.Categorical(coef_plot_data['Variable'], categories=list(reversed(list(var_ordered))), ordered=True)
from plotnine import ggplot, aes, geom_col, coord_flip, theme_classic

(ggplot(coef_plot_data, aes(x='Variable', y='Coefficient'))
+ geom_col()
+ coord_flip()
+ theme_classic()).save(filename="LogRegr_Coefficients.png", dpi=300)





#Save for later:


#Fit and Report function:
#Modified to be only for classification 
def fit_and_report_scores(model, X, y, Xv, yv):
    """
    Original source: 
    https://github.ubc.ca/MDS-2019-20/DSCI_573_feat-model-select_instructors/tree/master/source
    
    Original: fits a model and returns train and validation errors
    New: fits a model and returns accuracy, precision, and recall for train and validation sets
    Arguments
    ---------     
    model -- sklearn classifier model
        The sklearn model
    X -- numpy.ndarray        
        The X part of the train set
    y -- numpy.ndarray
        The y part of the train set
    Xv -- numpy.ndarray        
        The X part of the validation set
    yv -- numpy.ndarray
        The y part of the validation set       
    
    Keyword arguments 
    -----------------
    mode -- str 
        The mode for calculating error (default = 'regression') 
        TC: Changed default mode to classification for this analysis

    Returns
    -------
    errors -- list
        A list containing accuracy, precision, and recall on train (on X, y) and validation (on Xv, yv)
    
    """
    model.fit(X, y)
    pred_train = model.predict(X)
    pred_valid = model.predict(Xv)
    accuracy = [accuracy_score(y, pred_train), accuracy_score(yv, pred_valid)]
    precision = [precision_score(y, pred_train), precision_score(yv, pred_valid)]
    recall = [recall_score(y, pred_train), recall_score(yv, pred_valid)]
    scores = chain([accuracy, precision, recall])
    return scores


#model_results.columns = ["train_accuracy", "test_accuracy", 
#    "train_precision", "test_precision", 
#    "train_recall", "test_recall"]