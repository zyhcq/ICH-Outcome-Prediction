"""
00_train_location_classifier.py
Stage 0: Train cascaded XGBoost 5-class location model and generate CV/test probabilities.
"""
import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import sys

sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parent))
from config import (
    LOCATION_FEATURE_CSV, LOCATION_DIR, 
    LOCATION_TRAIN_CSV, LOCATION_TEST_CSV
)

warnings.filterwarnings('ignore')

class CascadeXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.l1_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss', use_label_encoder=False)
        self.l2a_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss', use_label_encoder=False)
        self.l2b_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss', use_label_encoder=False)

    def _map_label_l1(self, y):
        y_new = y.copy()
        y_new[np.isin(y, [0, 4])] = 0 # Deep
        y_new[np.isin(y, [1, 2])] = 1 # Infra
        y_new[y == 3] = 2             # Lobar
        return y_new

    def fit(self, X, y):
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y
        self.classes_ = np.unique(y_arr)
        
        y_l1 = self._map_label_l1(y_arr)
        self.l1_model.fit(X_arr, y_l1)
        
        mask_infra = np.isin(y_arr, [1, 2])
        if np.sum(mask_infra) > 0:
            X_infra = X_arr[mask_infra]
            y_infra = y_arr[mask_infra]
            y_infra_mapped = np.where(y_infra == 1, 0, 1)
            self.l2a_model.fit(X_infra, y_infra_mapped)
            
        mask_deep = np.isin(y_arr, [0, 4])
        if np.sum(mask_deep) > 0:
            X_deep = X_arr[mask_deep]
            y_deep = y_arr[mask_deep]
            y_deep_mapped = np.where(y_deep == 0, 0, 1)
            self.l2b_model.fit(X_deep, y_deep_mapped)
            
        return self

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, 'values') else X
        p_l1 = self.l1_model.predict_proba(X_arr) 
        
        p_l2a = self.l2a_model.predict_proba(X_arr) if hasattr(self.l2a_model, "classes_") else np.zeros((len(X_arr), 2))
        p_l2b = self.l2b_model.predict_proba(X_arr) if hasattr(self.l2b_model, "classes_") else np.zeros((len(X_arr), 2))
        
        p_final = np.zeros((len(X), 5))
        p_final[:, 0] = p_l1[:, 0] * p_l2b[:, 0]
        p_final[:, 1] = p_l1[:, 1] * p_l2a[:, 0]
        p_final[:, 2] = p_l1[:, 1] * p_l2a[:, 1]
        p_final[:, 3] = p_l1[:, 2]
        p_final[:, 4] = p_l1[:, 0] * p_l2b[:, 1]
        
        return p_final

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def main():
    if not LOCATION_FEATURE_CSV.exists():
        print(f"Error: {LOCATION_FEATURE_CSV} not found.")
        return
        
    df = pd.read_csv(LOCATION_FEATURE_CSV)
    df_train = df[df['subset'] == 'train'].reset_index(drop=True)
    df_test = df[df['subset'] == 'test'].reset_index(drop=True)
    
    drop_cols = ['id', 'subset', 'label']
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    
    X_train_raw = df_train[feature_cols]
    y_train = df_train['label'].astype(int)
    
    X_test_raw = df_test[feature_cols]
    y_test = df_test['label'].astype(int)
    
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=feature_cols)
    
    model = CascadeXGBClassifier()
    LOCATION_DIR.mkdir(parents=True, exist_ok=True)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    y_prob_cv = cross_val_predict(model, X_train_scaled, y_train, cv=skf, method='predict_proba', n_jobs=1)
    y_pred_cv = np.argmax(y_prob_cv, axis=1)
    
    df_train_cv = df_train[['id', 'label']].copy()
    df_train_cv['predict'] = y_pred_cv
    for i, class_name in enumerate(['Prob_Basal', 'Prob_Brainstem', 'Prob_Cerebellum', 'Prob_Lobar', 'Prob_Thalamus']):
        df_train_cv[class_name] = y_prob_cv[:, i]
    
    df_train_cv.to_csv(LOCATION_TRAIN_CSV, index=False)
    
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, LOCATION_DIR / 'cascaded_xgboost_model.pkl')
    joblib.dump(imputer, LOCATION_DIR / 'imputer.pkl')
    joblib.dump(scaler, LOCATION_DIR / 'scaler.pkl')
    
    y_prob_test = model.predict_proba(X_test_scaled)
    y_pred_test = np.argmax(y_prob_test, axis=1)
    
    df_test_res = df_test[['id', 'label']].copy()
    df_test_res['predict'] = y_pred_test
    for i, class_name in enumerate(['Prob_Basal', 'Prob_Brainstem', 'Prob_Cerebellum', 'Prob_Lobar', 'Prob_Thalamus']):
        df_test_res[class_name] = y_prob_test[:, i]

    df_test_res.to_csv(LOCATION_TEST_CSV, index=False)
    
    test_acc = accuracy_score(y_test, y_pred_test)
    test_b_acc = balanced_accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_acc:.4f}, Balanced Accuracy: {test_b_acc:.4f}")

if __name__ == "__main__":
    main()
