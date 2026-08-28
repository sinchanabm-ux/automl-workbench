<h1> AutoML-Workbench </h1>

<h2> A tool which autodetects the problem type, based on the uploaded CSV dataset, preprocesses, trains and compares models and shows the results in the web application. </h2>

So far:
- User can upload dataset
- User can select target variable via dropdown
- Data and Target Variable are separated into X and y respectively
- Unique-valued columns such as ID are dropped as encoding is not needed for them
- X undergoes three steps of pre-processing : Handling missing values, encoding, scaling

