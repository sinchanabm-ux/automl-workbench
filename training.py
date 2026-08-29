# Classification libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Regression libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score 

def train_models(X_train, X_test, y_train, y_test, problem_type) :
    results = {}
    if problem_type == "classification":
        # Logistic Regression
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        lr_y_pred = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_y_pred)
        results['Logistic Regression'] = lr_acc

        # Random Forest
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        rf_y_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_y_pred)
        results['Random Forest'] = rf_acc

    else:
        # Linear Regression
        li = LinearRegression()
        li.fit(X_train, y_train)
        li_y_pred = li.predict(X_test)
        li_acc = r2_score(y_test, li_y_pred)
        results['Linear Regression'] = li_acc

        # Random Forest Regressor
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        rf_y_pred = rf.predict(X_test)
        rf_acc = r2_score(y_test, rf_y_pred)
        results['Random Forest'] = rf_acc


    return results