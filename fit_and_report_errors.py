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
