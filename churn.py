# basics
import numpy as np
import pandas as pd
from itertools import chain


# classifiers / models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# other
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss, classification_report, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
import time

#Load data: 
from clean_load_data import clean_load_data
dat = clean_load_data()

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

from fit_and_report_errors import fit_and_report_errors
#Establish baseline/dummy model:
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy= "most_frequent")
dummy_errors = fit_and_report_errors(dummy, X_train, y_train, X_valid, y_valid)

#Set up dictionary for storing various models and their errors:
model_dict = {}
model_dict["dummy"] = dummy_errors

#Test basic models without hyper parameter optimization yet: 
max_iterations = 1000
models = {'Logistic regression': LogisticRegression(max_iter=max_iterations),
    'RBF SVM' : SVC(), 
    'random forest' : RandomForestClassifier() , 
    'neural net' : MLPClassifier(max_iter= max_iterations),
    'xgboost' : XGBClassifier(),
    'lgbm': LGBMClassifier()
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

#All models show an improvement over dummy model, with some (i.e., RBF SVM and Random Forest) showing HIGH signs of overfitting. 
#Next steps would be to re-run with some hyper parameter testing to see where I can find improvements 

grid_params = {'Logistic regression': {"C": 10.0 ** np.arange(-5,6), "solver": ["liblinear", "lbfgs"]},
    'RBF SVM' : {"C": 10.0 ** np.arange(-5,6)}, 
    'random forest' : {"max_depth" : np.arange(5,50,5), "max_leaf_nodes" : np.arange(30,150, 10)}, 
    'neural net' : {"activation" : ["relu"], "hidden_layer_sizes" : [(100,), (100, 10,), (100, 100, 10,), (100, 100, 100, 10,)]},
    'xgboost' : {"max_depth" : np.arange(5,50,5)},
    'lgbm': {"max_depth" : np.arange(5,50,5)} }

model_gs_dict = {}
models = {'Logistic regression': LogisticRegression(max_iter=max_iterations),'RBF SVM' : SVC()}

for model_name, model in models.items():
    start = time.time()
    param_grid = grid_params[model_name]
    gs_model = RandomizedSearchCV(model, param_grid)
    gs_model_result = gs_model.fit(X=X_train, y=y_train)
    end = time.time()
    time_to_model = round(end - start, 2)
    model_gs_dict[model_name] = {"params": gs_model_result.best_params_, #Save best parameters found
        "score": gs_model_result.best_score_, #Save best score
        "time": time_to_model, #Save model run time 
        "best_estimator": gs_model.best_estimator_} #Save best estimator for using for visualization
    print(model_name + " ran in " + str(time_to_model) + " seconds")

best_score = max(d['score'] for d in model_gs_dict.values())
for key in model_gs_dict.keys():
    d = model_gs_dict[key]
    if d['score'] == best_score:
        best_model = key
        break
final_model = model_gs_dict[best_model]["best_estimator"]
train_confusion_matrix = plot_confusion_matrix(final_model, X_train, y_train, values_format='.0f')
valid_confusion_matrix = plot_confusion_matrix(final_model, X_valid, y_valid, values_format='.0f')
import matplotlib.pyplot as plt
plt.show()

#What it means? 
#Simple way to see what's important is what features (that we have in the dataset) can help us understand customer churn beheaviour
#A simple way to do this is with Logistic Regression with the coefficients being an indication of feature importance
np.round(abs(final_model.coef_), decimals=2)
important_features = (np.round(abs(final_model.coef_), decimals=2) > 0.2).tolist()
coefs = final_model.coef_.tolist()[0]

#Make plot of coefficient importance: 
zipped_cols_coefs = zip(X_train.columns, coefs)
coef_plot_data = pd.DataFrame(zipped_cols_coefs)
coef_plot_data.columns = ["Variable", "Coefficient"]
coef_plot_data = coef_plot_data.sort_values('Coefficient')
#coef_plot_data["Variable_name"] = coef_plot_data["Variable"]

var_ordered = coef_plot_data['Variable'][coef_plot_data['Coefficient'].sort_values().index.tolist()] 
coef_plot_data['Variable'] = pd.Categorical(coef_plot_data['Variable'], categories=list(reversed(list(var_ordered))), ordered=True)


from plotnine import ggplot, aes, geom_col, coord_flip, theme_classic, scale_fill_continuous
(ggplot(coef_plot_data, aes(x='Variable', y='Coefficient', fill='Coefficient'))
+ geom_col()
+ coord_flip()
+ scale_fill_continuous()
+ theme_classic()).save(filename="LogRegr_Coefficients.png", dpi=300)

