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


# ---------------------------------------------------------------------------
# Cleaning for the downloadable CSV (independent of any target column)
# ---------------------------------------------------------------------------
HIGH_MISSING_RATIO = 0.8   # same threshold preprocess_data() uses

def clean_data(df):
    """Clean a dataset for download. Returns (cleaned_df, report).

    Rules (kept human-readable: no one-hot encoding, no scaling):
      1. remove exact duplicate rows
      2. drop columns where 80% or more of the values are missing
      3. fill remaining missing values: median for numbers, most frequent value otherwise
    Raises ValueError if nothing usable is left.
    """
    rows_before, cols_before = df.shape
    cleaned = df.drop_duplicates()
    duplicates_removed = rows_before - len(cleaned)

    if len(cleaned) == 0:
        raise ValueError("The dataset has no rows.")

    missing_ratio = cleaned.isnull().mean()
    dropped_columns = missing_ratio[missing_ratio >= HIGH_MISSING_RATIO].index.to_list()
    cleaned = cleaned.drop(columns=dropped_columns)

    if cleaned.shape[1] == 0:
        raise ValueError("No usable columns remain: too many values are missing.")

    filled_by_column = {}
    for col in cleaned.columns:
        n_missing = int(cleaned[col].isnull().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[col]) and cleaned[col].dtype != 'bool':
            fill_value = cleaned[col].median()
        else:
            mode = cleaned[col].mode()
            if mode.empty:
                continue
            fill_value = mode[0]
        cleaned[col] = cleaned[col].fillna(fill_value)
        filled_by_column[col] = n_missing

    missing_filled = sum(filled_by_column.values())
    report = {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "columns_before": cols_before,
        "columns_after": cleaned.shape[1],
        "duplicates_removed": duplicates_removed,
        "dropped_columns": dropped_columns,
        "missing_filled": missing_filled,
        "filled_by_column": filled_by_column,
        "changes_made": bool(duplicates_removed or dropped_columns or missing_filled),
    }
    return cleaned, report