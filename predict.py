import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score


def predict(data_path, pkl_path, ohe=None):
    from sklearn.preprocessing import StandardScaler
    """Predict enhancer data.

    Parameters
    ----------
    data_path : str
        Path to data directory, e.g., "X_train.csv" or "X_test.csv".
    pkl_path : str
        Path to additional required data. Here, it's the weights of the torch model, "model.pkl".
    ohe : OneHotEncoder
        Used to ensure categories are the same in train/test.
        Must set handle_unknown='ignore'. Required at test time.
        Include this in your pkl and unpack it in the predict.py script.

    Returns
    -------
    df_pred : pd.DataFrame
        Prediction formatted as required by the score function.
    """
    
    # Cleaning and feature engineering input data
    input_data = pd.read_csv(data_path)
    features = ['distanceToTSS', 'numTSSEnhGene',
                'numCandidateEnhGene', 'normalizedDNase_enh', 'normalizedDNase_prom',
                'numNearbyEnhancers', 'sumNearbyEnhancers', 'ubiquitousExpressedGene',
                '3DContact', '3DContact_squared', 'normalizedDNase_enh_squared',
                'ABC.Score']

    input_data = input_data.fillna(0)
    X_train = input_data[features]
    
    #Scaling
    float_columns = ["ABC.Score", "normalizedDNase_enh_squared", "3DContact_squared", "3DContact"]
    X_float = input_data[float_columns].copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(X_float)
    input_data.loc[:, float_columns] = scaled_values
    
    #loading in pickle files
    with open(pkl_path, 'rb') as f:
        loaded_models = pickle.load(f)
    
    #array of predictions
    probas = []
    
    #predict for everyrow
    for _, row in input_data.iterrows():
        chr_tss = row['chrTSS']
        if chr_tss in ['chr4', 'chr14', 'chr18', 'chr21']:
            model = loaded_models['xgb_model_sig']
          
        elif chr_tss in ['chr5', 'chr13']:
            model = loaded_models['logistic_regression_reg']
           
        else:
            model = loaded_models['xgb_model_reg']
            
        proba = model.predict_proba(row[features].values.reshape(1, -1))[:, 1][0]

        probas.append(proba)
 
    
    return probas