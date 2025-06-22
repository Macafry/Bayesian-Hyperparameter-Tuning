import os, shutil
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

import xgboost as xgb
import lightgbm as lgb
import catboost as cat

from sklearn.metrics import roc_auc_score

def predict_xgb(model, X):
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)
    
def predict_lgb(model, X):
    return model.predict(X, num_iteration=model.best_iteration)

def predict_cat(model, X):
    return model.predict_proba(X)[:, 1]


def load_model(model_path):
    if model_path.endswith("json"):
        model = xgb.Booster()
        model.load_model(model_path)
        
        predict = lambda X: predict_xgb(model, X)
        
        
    elif model_path.endswith("txt"):
        model = lgb.Booster(model_file=model_path)
        
        predict = lambda X: predict_lgb(model, X)
        
    elif model_path.endswith("cbm"):
        model = cat.CatBoostClassifier()
        model.load_model(model_path)
        
        predict = lambda X: predict_cat(model, X)
    else:
        raise ValueError("Unknown model type")
    
    return model, predict


def predict_and_save_in_chunks(parquet_path, model_precict, output_dir, label_col = "target"):
    
    chunks_path = os.path.join(output_dir, 'chunks')
    os.makedirs(chunks_path, exist_ok=True)
    
    
    reader = pq.ParquetFile(parquet_path).iter_batches(batch_size=100_000)
    
    for i, chunk in enumerate(reader):
        X = chunk.to_pandas().drop(columns=[label_col])
        y_true = chunk.to_pandas()[label_col]
        y_pred = model_precict(X)

        out_chunk = pl.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred
        })

        fpath = os.path.join(chunks_path, f'chunk_{i}.parquet')
        out_chunk.write_parquet(fpath)
        
    df = pl.scan_parquet(os.path.join(chunks_path, "*.parquet"))
    df.collect().write_parquet(os.path.join(output_dir, "predictions.parquet"))
    
    shutil.rmtree(chunks_path)
    

def auc_from_file(parquet_path, model_path):
    _, predict = load_model(model_path)
    
    # Get file name and model name
    file_name = os.path.splitext(os.path.basename(parquet_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Build output directory path and predictions file path
    out_dir = os.path.join('test', model_name, file_name)
    preds_path = os.path.join('test', model_name, file_name, 'predictions.parquet')
    
    # Save predictions to file
    predict_and_save_in_chunks(parquet_path, predict, out_dir)
    
    # Calculate AUC and return
    df = pl.read_parquet(preds_path).to_pandas()
    auc = roc_auc_score(df['y_true'], df['y_pred'])
    
    return {
            'model': model_name,
            'data': file_name,
            'auc': auc,
    }
    
def evaluate_all_models(models_path, parquet_files):
    # get train/test predictions for all models
    aucs = list()
    for model in os.listdir(models_path):
        model_path = os.path.join(models_path, model)
        
        for file in parquet_files:
            aucs.append(auc_from_file(file, model_path))

    df = pd.DataFrame(aucs)
    df.to_csv('higgs_data/aucs.csv', index=False)

    # clean up
    shutil.rmtree('./test')
    
    return df