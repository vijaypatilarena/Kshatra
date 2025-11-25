import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cProfile import label
import joblib
import numpy as np
from src import features
from src.screening.refine import closest_approach, _pos_vel_from_sat
from src.features.encounter_feature  import pair_features

MODEL_PATH= "models/collision_predictor_xgboost.pkl"

def predict_from_tles(A,B):
    #Finding the closest approach
    tca_jd, miss_km, vrel = closest_approach(A, B)
    
    #Extracting the features
    rA, vA = _pos_vel_from_sat(A, tca_jd)
    rB, vB = _pos_vel_from_sat(B, tca_jd)

    feats = pair_features(rA, vA, rB, vB)
    feats["tca_jd"] = tca_jd
    feats["miss_km"] = miss_km
    feats["vrel_kms"] = vrel

    x = np.array([list(feats.values())])
    prob = model.predict_probs(x)[0][1]

    label = int(prob>= 0.5)

    return{
        "label": label,
        "probability": prob,
        "miss_km": miss_km,
        "vrel_kms": vrel,
        "tca_jd": tca_jd,
        "features": feats,
        
    }