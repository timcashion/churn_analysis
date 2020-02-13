import pandas as pd
from sklearn.preprocessing import StandardScaler
def clean_load_data():
    dat = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    dat.head()
    dat.describe()
    dat = dat.drop(columns=["customerID"])

    #Clean datasets and encode where necessary:
    dat.Churn = dat.Churn.map(dict(Yes=1, No=0))

    #Filter out " " Total Charges:
    dat = dat[dat.TotalCharges != " "]


    #dat.dtypes #Check data types for cleaning and encoding
    
    numeric_cols = ["MonthlyCharges", "TotalCharges"]
    #Scale numeric columns to deal with wide data distribution: 
    dat[numeric_cols] = dat[numeric_cols].astype(float)
    scaler = StandardScaler()
    dat[numeric_cols] = scaler.fit_transform(dat[numeric_cols])
    
    #Categorical/ already encoded columns. 
    integer_cols = ["SeniorCitizen", "tenure"]
    categorical_cols = ["gender", 'Partner', "Dependents", "PhoneService", 
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"]
    dat = pd.get_dummies(dat, columns=categorical_cols, drop_first=True)
    dat[integer_cols] = dat[integer_cols].astype(int)
    return dat




