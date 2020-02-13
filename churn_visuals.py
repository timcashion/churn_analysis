




#Do we need to normalize our numeric and integer cols?
#SeniorCitizen was already in binary form. So no. 

from plotnine import ggplot, aes, geom_histogram, geom_boxplot
(ggplot(dat, aes(x='MonthlyCharges'))
+ geom_histogram()).save(filename="MonthlyCharges_Hist.png", dpi=300)

(ggplot(dat, aes(x='TotalCharges'))
+ geom_histogram()).save(filename="TotalCharges_Hist.png", dpi=300)

#Neither follow a normal distribution. Log transformation could help, but these are odd. 
dat["LogTotalCharges"] = np.log(dat["TotalCharges"]+1)
dat["LogMonthlyCharges"] = np.log(dat["MonthlyCharges"]+1)


(ggplot(dat, aes(x='LogMonthlyCharges'))
+ geom_histogram())

(ggplot(dat, aes(x='LogTotalCharges'))
+ geom_histogram())

#Doesn't really help so leave this for now. 

dat = dat.drop(columns = ["LogTotalCharges", "LogMonthlyCharges"])



dat["Churn_label"] = dat["Churn"].astype(str)

(ggplot(dat, aes(x="Churn_label", y='MonthlyCharges'))
+ geom_boxplot()).save(filename="MonthlyChargesChurn_Box.png", dpi=300)

(ggplot(dat, aes(x="Churn_label", y='TotalCharges'))
+ geom_boxplot()).save(filename="TotalChargesChurn_Box.png", dpi=300)

dat = dat.drop(columns="Churn_label")