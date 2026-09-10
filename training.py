from sklearn.model_selection import cross_val_score
# Classification libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regression libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_models(X, y, problem_type) :
    results = {}
    if problem_type == "classification":
        # Logistic Regression
        lr = LogisticRegression()
        lr_scores = cross_val_score(lr, X, y, cv=5) # 5 folds
        lr_mean_score = lr_scores.mean()
        lr.fit(X, y)
        results['Logistic Regression'] = lr_mean_score

        # Random Forest
        rf = RandomForestClassifier()
        rf_scores = cross_val_score(rf, X, y, cv=5)
        rf_mean_score = rf_scores.mean()
        rf.fit(X, y)
        results['Random Forest'] = rf_mean_score

    else:
        # Linear Regression
        li = LinearRegression()
        li_scores = cross_val_score(li, X, y, cv=5)
        li_mean_score = li_scores.mean()
        li.fit(X, y)
        results['Linear Regression'] = li_mean_score

        # Random Forest Regressor
        rf = RandomForestRegressor()
        rf_scores = cross_val_score(rf, X, y, cv=5)
        rf_mean_score = rf_scores.mean()
        rf.fit(X, y)
        results['Random Forest'] = rf_mean_score


    return results