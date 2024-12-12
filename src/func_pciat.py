import pandas as pd
import time
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# TEMP Define model 

# First define PCIAT-sii mapping function: 

def cat_sii(val):
    if 0 <= val <= 30:
        return 0.0
    elif 31 <= val <= 49:
        return 1.0
    elif 50 <= val <= 79:
        return 2.0
    else: 
        return 3.0

# Define function for PCIAT-sii prediction:

def pciat_sii_pred(data, model):
    """Function which uses "model" to train to predict each PCIAT value and then maps them to sii values. Note:
    due to speicificities of the approach, the data provided needs to be BEFORE train-test split, the function
    will execute the split itself. It outputs y_pred, and y_test # For now PCIAT pred

    Args:
        data (_type_): dataframe with data
        model (_type_): initialised model object

    Returns:
       y_test, y_pred -> a dataframe with predicted values, as well as a dataframe with test values which can then be assessed, e.g. with an f1 metric.
    """
    time1 = time.time()

    # Define the PCIAT columns
    pciat_vals = ['PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04',
       'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08',
       'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12',
       'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16',
       'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20']
    
    # Train test split
    
    train, test = train_test_split(data, test_size=0.3, random_state=42)

    # Drops NAs of PCIAT form train & test

    train1 = train.dropna(subset=pciat_vals)
    test1 = test.dropna(subset=pciat_vals)

    # Drop sii columns, include only float and integer values 

    train2 = train1.drop("sii", axis=1).select_dtypes(include=['float64', 'int64'])
    test2 = test1.drop("sii", axis=1).select_dtypes(include=['float64', 'int64'])

    # Initialise a df where we store the predicted values

    pciat_pred = pd.DataFrame(index=range(test1.shape[0]))

    # Loop over each PCIAT, predicting it
    for val in pciat_vals:
        other_vals = [x for x in pciat_vals if x != val] # Make list for remaining values for filtering
        # print(other_vals)
        train3 = train2.drop(other_vals, axis=1) # Drop other PCIATs so that it can't be predicted from them
        test3 = test2.drop(other_vals, axis=1) 
        X_train = train3.drop(val, axis=1) # do the X y split
        X_test = test3.drop(val, axis=1)
        y_train = train3[val]
        y_test = test3[val]
        model.fit(X_train, y_train) # Train the model 
        y_pred = model.predict(X_test) # predict the PCIAT values
        pciat_pred[val] = y_pred # update the PCIAT prediction
    
    # Calculate the total column
    pciat_pred["total"]=pciat_pred[pciat_pred.columns].sum(axis=1) 

    #Map the total column to sii prediction

    y_pred = pciat_pred['total'].apply(cat_sii)

    # Define y_true 
    
    y_test = test1['sii']

    time2 = time.time()
    print(f"time to run: {time2-time1} sec")
    return y_test, y_pred

