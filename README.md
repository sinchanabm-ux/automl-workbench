<h1> AutoML-Workbench </h1>

<h2> A tool which autodetects the problem type, based on the uploaded CSV dataset, preprocesses, trains and compares models and shows the results in the web application. </h2>

So far:
- User can upload dataset
- User can select target variable via dropdown
- Data and Target Variable are separated into X and y respectively
- User finds out whether the dataset requires classification or Regression based on Target Variable
- Unique-valued columns such as ID are dropped as encoding is not needed for them
- Mostly empty / null-valued columns are dropped
- X undergoes three steps of pre-processing : Handling missing values, encoding, scaling
- Classification offers two models : Logistic Regression and Random Forest
- Regression offers two models : Linear Regression and Random Forest 

