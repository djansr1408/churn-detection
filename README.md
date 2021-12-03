# Churn detection on telco customers dataset

The repository contains notebook and data for building, training and evaluating ML classification model for churn detection.
Download it using *git clone* command and navigate there:
```
git clone https://github.com/djansr1408/churn-detection.git
cd churn-detection
```
In order to run the notebook, it is necessary to set up Python environment first. Here are the commands to do so:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Run *jupyter* using the command:
```
jupyter notebook
```
Open [data_analysis.ipynb](data_analysis.ipynb) notebook and run **Kernel/Restart and run all**.

[data_analysis.ipynb](data_analysis.ipynb) notebook consist of a few parts and each of them is followed with corresponding explanation.
- **Data exploration**: Includes data reading, checking for duplicates, checking datatypes, counting distinct values per each categorical column.
- **Feature creation**: Preprocessing numerical and categorical columns. Encoding categorical values, pivoting categorical columns with multiple values, merging all these into one final dataset.
- **Feature correlation**: Inspecting correlation between features, so to understand relations between them.
- **Train test split**: Spliting the dataset to train and test, where train dataset will be used for cross-validation and test dataset for evaluation.
- Scaler: MinMaxScaler used for scaling numerical features, categorical features are already in range [0, 1].
- Hyperparameter optimization: Params distributions are read from *./storage* directory and for each model and each combination of hyperparameters cross-validation is performed. At the end, the best model with optimal hyperparameters is recorded.
- Model: Training and validation of the best model. In this case, it was logistic regression.
- Metrics: Includes confussion matrix, ROC curve, Precision-Recall Curve.
- Weighted logistic regression: Training and evaluation of the weighted logistic regression model. Includes confussion matrix, ROC curve, PR curve.
- Feature importances: Visualization of feature importances of the weighted logistic regression model.
- TSNE Approach: Customer visualization in 2d using TSNE model.
- Churners good and bad examples: Manual inspection of False Positive, False Negative and True Positive examples using [LIME](https://github.com/marcotcr/lime)
