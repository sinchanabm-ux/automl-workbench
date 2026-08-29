import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column):
    #separate target column
    X = df.drop(columns=[target_column])
    y = df[target_column]

    #Detection Logic(classification / regression)
    if y.dtype == 'object' or y.nunique() <= 10:
        problem_type  = "classification"
    else:
        problem_type = "regression"

    #checking for columns with more than 80% of missing values and dropping them
    high_missing = [] 
    for col in X.columns: 
        ratio_miss = X[col].isnull().sum()/len(X)
        if ratio_miss >= 0.8:
            high_missing.append(col)
    X = X.drop(columns = high_missing)
    #if all the columns in the dataset had more than 80% missing values
    if X.shape[1] == 0:
        return None, None, None, None, None, "Not enough usable dataset remains after cleaning as too many columns had excessive missing values"

    #separate numerical and categorical columns into two lists
    num = X.select_dtypes(include='number').columns.to_list()
    obj = X.select_dtypes(include= ['object', 'str']).columns.to_list()

    #checking for ID columns which are to be dropped as encoding is not needed for columns with completely unique values
    ID = []
    for col in obj:
        ratio_ID = X[col].nunique()/len(X)
        if ratio_ID >= 0.9:
            ID.append(col)
    X = X.drop(columns = ID)
    for col in ID:
        obj.remove(col)
    
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

    return X_train, X_test, y_train, y_test, problem_type, None