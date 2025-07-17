import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Function to format a 2D array into a CSV-like string
def formatArray2D(arr):
    output = ""
    for i_row, row in enumerate(arr):
        for i, val in enumerate(row):
            if i != 0 and i_row != 0:
                val = round(float(val), 4)  # Round values to 4 decimal places for readability
            output += str(val)
            if i < len(row) - 1:
                output += ","  # Add a comma between values
        output += "\n"  # Newline at the end of each row
    return output

# Function to create a KNN pipeline and parameter grid
def KNN(seed, smote_params):
    pipeline = Pipeline([('smote', SMOTE(random_state=seed)),
                         ('knn', KNeighborsClassifier())])
    param_grid = [{'knn__n_neighbors': [90, 100, 110],
                   'knn__weights': ['uniform'],
                   'knn__algorithm': ['ball_tree'],
                   'knn__leaf_size': [3, 5, 7],
                   **smote_params}]
    return pipeline, param_grid

# Function to create a Logistic Regression pipeline and parameter grid
def LR(seed, smote_params):
    pipeline = Pipeline([('smote', SMOTE(random_state=seed)),
                         ('lr', LogisticRegression(solver='saga', random_state=seed, max_iter=100))])
    param_grid = [{'lr__penalty': ['l1', 'l2'],
                   'lr__C': np.power(10.0, np.arange(-2, 2, 1)),
                   **smote_params}]
    return pipeline, param_grid

# Function to create an SVM pipeline and parameter grid
def RBFSVM(seed, smote_params):
    pipeline = Pipeline([('smote', SMOTE(random_state=seed)),
                         ('rbfsvm', SVC(kernel='rbf', probability=True, random_state=seed))])
    param_grid = [{'rbfsvm__C': [0.001],
                   'rbfsvm__gamma': np.power(10., np.arange(-3, 0, 0.5)),
                   'rbfsvm__tol': np.power(10., np.arange(-10, -6, 1)),
                   **smote_params}]
    return pipeline, param_grid

# Function to create a Random Forest pipeline and parameter grid
def RF(seed, smote_params):
    pipeline = Pipeline([('smote', SMOTE(random_state=seed)),
                         ('rf', RandomForestClassifier(random_state=seed))])
    param_grid = [{'rf__max_depth': [1, 3, 5],
                   'rf__n_estimators': [50, 100, 250],
                   'rf__criterion': ['gini'],
                   'rf__max_features': ['sqrt'],
                   **smote_params}]
    return pipeline, param_grid

# Function to create an AdaBoost pipeline and parameter grid
def AB(seed, smote_params):
    pipeline = Pipeline([('smote', SMOTE(random_state=seed)),
                         ('ab', AdaBoostClassifier(random_state=seed))])
    param_grid = [{'ab__learning_rate': [0.005, 0.01, 0.05],
                   'ab__n_estimators': [10, 20, 30, 100],
                   **smote_params}]
    return pipeline, param_grid

# Function to create a Gradient Boosting pipeline and parameter grid
def GB(seed, smote_params):
    pipeline = Pipeline([('smote', SMOTE(random_state=seed)),
                         ('gb', HistGradientBoostingClassifier(random_state=seed))])
    param_grid = [{'gb__max_iter': [50, 75, 100],
                   'gb__learning_rate': [0.01],
                   'gb__min_samples_leaf': [1, 3, 5],
                   'gb__max_depth': [1, 3, 5],
                   **smote_params}]
    return pipeline, param_grid

# List of random seeds to ensure reproducibility
seeds = [685, 136, 149, 946, 422, 617, 822, 847, 64, 415]

# Load the dataset
data = pd.read_csv("data/Data.csv", sep=',')
print(data.columns)

# Split features and target
X = data.drop(['Prop', 'Prop_class'], axis=1)
yy = pd.DataFrame(data[['Prop', 'Prop_class']])

# Define the models and their names
MODELS_FUNC = [KNN, LR, RBFSVM, RF, AB, GB]
MODELS_NAME = ["KNN", "LR", "RBFSVM", "RF", "AB", "GB"]

# Initialize lists to store results
result_train = [[] for _ in range(len(MODELS_FUNC))]
result_test = [[] for _ in range(len(MODELS_FUNC))]

for seed in seeds:
    # Split the data into train and test sets
    X_train, X_test, yy_train, yy_test = train_test_split(X, yy, test_size=0.2, random_state=seed)

    # Extract target variables
    y_prop = yy["Prop"]
    y_train = yy_train["Prop_class"]
    y_test = yy_test["Prop_class"]

    # Prepare final training and test sets
    X_train_final = X_train.copy()
    X_test_final = X_test.copy()

    y_train_final = y_train.copy()
    y_test_final = y_test.copy()
    
    # Initialize stratified k-fold cross-validation
    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    # Define SMOTE parameters
    smote_params = {'smote__sampling_strategy': [0.5, 0.7, 0.9, 'auto'],
                    'smote__k_neighbors': [3, 5, 7, 9]}

    # Iterate over each model
    for i_model, pipe in enumerate(MODELS_FUNC):
        # Get pipeline and parameter grid for the current model
        pipeline, param_grid = pipe(seed, smote_params)

        # Perform grid search with cross-validation
        gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0, scoring='f1_macro')
        gs.fit(X_train_final, y_train_final)

        # Predict on training and test sets
        pred_train = gs.predict(X_train_final)
        pred_test = gs.predict(X_test_final)

        # Generate classification reports
        f1_train_report = classification_report(y_train_final, pred_train, output_dict=True)
        f1_test_report = classification_report(y_test_final, pred_test, output_dict=True)

        # Convert classification reports to DataFrames
        dict_train = pd.DataFrame.from_dict(f1_train_report)
        dict_test = pd.DataFrame.from_dict(f1_test_report)
        f1_matrix_train = pd.concat([dict_train.transpose()['f1-score']], axis=1)
        f1_matrix_test = pd.concat([dict_test.transpose()['f1-score']], axis=1)

        # Collect f1-scores for each seed
        list_seed_train = []
        list_seed_test = []
        for i_value in range(5):
            list_seed_train.append(f1_matrix_train.iloc[i_value].values[0])
            list_seed_test.append(f1_matrix_test.iloc[i_value].values[0])
        
        result_train[i_model].append(list_seed_train)
        result_test[i_model].append(list_seed_test)

# Create a summary table for results
table = [["Model", "Cat0_train", "Cat1_train", "<Acc>_train", "<MacroAvg>_train", "<WeightAvg>_train",
          "Cat0_test", "Cat1_test", "<Acc>_test", "<MacroAvg>_test", "<WeightAvg>_test"]]

# Calculate average metrics for each model
for i_model, model in enumerate(MODELS_NAME):
    row_table = [model]
    for result in [result_train, result_test]:
        sum_values = np.zeros(5)
        for i_seed, seed in enumerate(seeds):
            for i_value in range(5):
                sum_values[i_value] += float(result[i_model][i_seed][i_value])
        for i_value in range(5):
            row_table.append(round(sum_values[i_value] / len(seeds), 4))
    table.append(row_table)

# Save the summary statistics to a CSV file
with open("Results.csv", "w") as file:
    file.write(formatArray2D(table))