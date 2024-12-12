# The Kaggle ChildMind project
https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use

## PCIAT -> sii function: 
The target variable, sii, is derived directly from PCIAT_Total, which is a sum of the 20 PCIAT columns (questionnaire answers).

The wrapper function `pciat_sii_pred` was written, which allows to run an ML model to predict each of the 20 PCIAT values and map them to `sii`, rather than predicting sii directly. This was shown to improve metrics (accuracy, f1 score) for a basic decision tree model. 

To run function, make sure the `func_pciat.py` file is in your directory, and import the function from there (the file is in the `src` folder). Then initialise a model object (eg with scikit-learn), and supply the training data as well as the model to  `pciat_sii_pred` function :). NOTE THAT THE MODEL DOES THE TRAIN TEST SPLIT ITSELF (see details below on reason for this in details). For example:

```python 
from func_pciat import pciat_sii_pred
train = pd.read_csv('../data/train.csv')
model = DecisionTreeClassifier()

y_pred, y_test = pciat_sii_pred(train, model)

```

Some important notes on this: 
- The function does the train test split itself. This is because the train / test split needs to be done BEFORE x / y (ie features and targets) split, to ensure that PCIAT values are predicted for the same set of observations, so that they can be re-combined to yield our sii prediction.  
- The function doesn't (yet?) work for cross-validation or where multiple models are trained as part of same initialised model object. 


More info can be found in docstrings of the function.