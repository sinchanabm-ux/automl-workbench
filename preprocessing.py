import pandas as pd
from sklearn.preprocessing import StandardScaler

# Categorical columns with more unique values than this are dropped instead of
# one-hot encoded (keeps memory usage safe on Render's free tier)
MAX_CATEGORIES = 50

def preprocess_data(df, target_column):
    #rows with a missing target can't be used for training, so drop them
    df = df.dropna(subset=[target_column])
    if len(df) == 0:
        return None, None, None, "The selected target column has no values."

    #separate target column
    X = df.drop(columns=[target_column])
    y = df[target_column]

    #Detection Logic(classification / regression)
    #non-numeric targets (text, in any pandas version) and booleans are always classification
    if (not pd.api.types.is_numeric_dtype(y)) or y.dtype == 'bool' or y.nunique() <= 10:
        problem_type = "classification"
    else:
        problem_type = "regression"

    if problem_type == "classification" and y.nunique() < 2:
        return None, None, None, "The target column has only one distinct value, so there is nothing to predict."

    #checking for columns with more than 80% of missing values and dropping them
    high_missing = [] 
    for col in X.columns: 
        ratio_miss = X[col].isnull().sum()/len(X)
        if ratio_miss >= 0.8:
            high_missing.append(col)
    X = X.drop(columns = high_missing)
    #if all the columns in the dataset had more than 80% missing values
    if X.shape[1] == 0:
        return None, None, None, "Not enough usable dataset remains after cleaning as too many columns had excessive missing values"

    #separate numerical and categorical columns into two lists
    num = X.select_dtypes(include='number').columns.to_list()
    obj = X.select_dtypes(include= ['object', 'str', 'category']).columns.to_list()

    #dropping ID-like columns (almost all values unique) and columns with too many
    #categories to one-hot encode safely
    ID = []
    for col in obj:
        ratio_ID = X[col].nunique()/len(X)
        if ratio_ID >= 0.9 or X[col].nunique() > MAX_CATEGORIES:
            ID.append(col)
    X = X.drop(columns = ID)
    for col in ID:
        obj.remove(col)

    if X.shape[1] == 0:
        return None, None, None, "No usable feature columns remain after dropping ID-like and high-cardinality columns."
    
    #Step-1 of preprocessing : Handle missing values, median for numerical data, mode for categorical data
    for col in num:
        median = X[col].median()
        X[col] = X[col].fillna(median)
    for col in obj:
        mode = X[col].mode()[0]
        X[col] = X[col].fillna(mode)

    #Step-2 of preprocessing : Encoding categorical data 
    X = pd.get_dummies(X, columns= obj)

    #Step-3 of preprocessing : Scaling numerical columns
    if num: 
        scaler = StandardScaler()
        X[num] = scaler.fit_transform(X[num])


    return X, y, problem_type, None