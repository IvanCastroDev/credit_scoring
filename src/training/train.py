import os
import sys
import yaml
import math
import torch
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
import mlflow.pytorch
import torch.nn as nn
import logging as log
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.nn import Module
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)
from mlflow.models.signature import infer_signature

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.processing.main import CreditDataProcessor
from src.training.model import CreditScoringModel

def setup_logging(level=log.INFO, log_file: str | None = None):
    handlers = [log.StreamHandler(sys.stdout)] # stdout para imprimir en consola

    if log_file:
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")) # Para guardar los logs en un archivo

    log.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True
    )

    # Bajamos el ruido de las librearias de terceros
    for noisy in ('mlflow', 'urlib3', 'matplotlib'):
        log.getLogger(noisy).setLevel(log.WARNING)

class CreditScoringModelTraining:
    def __init__(self, config_path: Path) -> None:
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        log.info(f"--- Config Training ---")

        # paths -------------------------------------
        self.dataset_path = Path(self.params['data_source']['data_path']['dataset_path'])
        self.artifact_name_or_path = self.params['data_source']['data_path']['artifact_path']
        self.preprocessor_filename = self.params['data_source']['data_path']['preprocessor_filename']
        
        # architecture -------------------------------------
        model_cfg = self.params['model_config']['architecture']
        self.hidden_layers = model_cfg['hidden_layers']
        self.use_batch_norm = model_cfg['use_batch_norm']
        self.activation_fn = model_cfg['activation_fn']
        self.dropout_rate = model_cfg['dropout_rate']
        
        # training -------------------------------------
        train_cfg = self.params['training_params']
        self.optimizer_name = train_cfg['optimizer']['name']
        self.learning_rate = train_cfg['optimizer']['learning_rate']
        self.weight_decay = train_cfg['optimizer'].get('weight_decay', 0.0) # Default a 0
        self.use_pos_weight = train_cfg['loss_function']['use_pos_weight']
        self.scheduler_patience = train_cfg['scheduler']['patience']
        self.scheduler_factor = train_cfg['scheduler']['factor']
        self.epochs = train_cfg['epochs']
        self.batch_size = train_cfg['batch_size']
        
        # data -------------------------------------
        self.test_size = train_cfg['test_size']
        self.random_state = train_cfg['random_state']
        
        # early stopping -------------------------------------
        self.early_stopping_patience = train_cfg['early_stopping']['patience']
        self.early_stopping_delta = train_cfg['early_stopping']['delta']
        
        # Model an project name -------------------------------------
        self.model_name = self.params['model_config']['model_name']
        self.mlflow_project_name = self.params['mlflow_config']['mlflow_project_name']

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # Set device (Trabajar con los nucleos cuda si se encuentran disponibles)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_processor = CreditDataProcessor()

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_auc": [],
            "val_auc": []
        }

        self.local_artifacts_dir = Path("reports")
        self.local_artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            log.error(f'File not found at: {self.dataset_path}')
            raise

        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'])

        df_train, df_val = train_test_split(
            df,
            test_size = self.test_size,
            random_state= self.random_state,
            stratify=df[self.data_processor.targe_feature]
        )

        return df_train, df_val

    def _process_data(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[torch.Tensor, ...]:
        """
        Fits preprocessor on training data and transforms both sets
        """

        log.info("Processing data...")
        preprocessor = self.data_processor.fit_preprocessor(df_train)

        x_train_processed, y_train_processed = self.data_processor.process_data(df_train, preprocessor)
        x_val_processed, y_val_processed = self.data_processor.process_data(df_val, preprocessor)

        print(f"Processed dataframes: {x_train_processed, y_train_processed.head()}")

        x_train_tensor = torch.tensor(x_train_processed, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_processed.values, dtype=torch.float32).view(-1, 1)
        x_val_tensor = torch.tensor(x_val_processed, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_processed.values, dtype=torch.float32).view(-1, 1)

        path_processor = f"models/{self.preprocessor_filename}"
        joblib.dump(preprocessor, path_processor)
        log.info(f"Processor saved to {path_processor}")

        return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor
    
    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float | Any]:
        """
        Calc accuracy, precision, recall f1 and ROC-AUC
        Args:
            y_true (np.ndarray): Real categories
            y_prob (np.ndarray): Predicted categories
        """
        y_pred = (y_prob >= threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred)

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float('nan')
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        }
    
    def _evaluate_split(self, model: CreditScoringModel, x: torch.Tensor, y: torch.Tensor, criterion: Module) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y).item()
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            y_true = y.detach().cpu().numpy().reshape(-1)
            m = self._compute_metrics(y_true, prob, threshold=0.5)
            m['loss'] = loss

        return m
    
    def _plot_and_save(self, xs: List[int], ys1: List[float], ys2: List[float], title: str, ylabel: str, filename: str):
        plt.figure()
        plt.plot(xs, ys1, label="train")
        plt.plot(xs, ys2, label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        out = self.local_artifacts_dir / filename
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return out
    
    def _plt_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, filename: str):
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix (val)")
        plt.colorbar()
        tick_marks = [0,1]
        plt.xticks(tick_marks, ['bad(0)', 'good(1)'])
        plt.yticks(tick_marks, ['bad(0)', 'good(1)'])

        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha='center', va='center',
                         color='white' if cm[i, j] > thresh else 'black')
                
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        out = self.local_artifacts_dir / filename
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        return out
    
    def _plt_roc_pr(self, y_true: np.ndarray, y_prob: np.ndarray, roc_file: str, pr_file: str):
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.plot([0,1],[0,1], linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (val)")
            roc_path = self.local_artifacts_dir / roc_file
            plt.savefig(roc_path, bbox_inches="tight")
            plt.close()
        except ValueError:
            roc_path = None

        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve (val)")
            pr_path = self.local_artifacts_dir / pr_file
            plt.savefig(pr_path, bbox_inches="tight")
            plt.close()
        except:
            pr_path = None

        return roc_path, pr_path
    
    def _run_trainig_loop(self, model: CreditScoringModel, criterion: Module, optimizer: optim, scheduler, x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame, y_val: pd.DataFrame):
        """
        Executes the main train and validation loop with early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        epochs_run = 0

        log.info('Starting training loop -----------------------')
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0
            epochs_run = epoch + 1
            
            # Mini batches
            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()

if __name__ == "__main__":
    setup_logging(log_file='logs/log.txt')
    
    parser = argparse.ArgumentParser(description="Threshold optimizer for ID classification model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/credit_scoring-training_config-german_credit_risk_v110.yaml",
        help="Path to the threshold optimizer YAML config."
    )

    cli_args = parser.parse_args()

    log.info(f"Config path: {cli_args.config}")

    try:
        trainer = CreditScoringModelTraining(cli_args.config)
        df_test, df_val = trainer._load_and_split_data()

        trainer._process_data(df_test, df_val)
    except Exception as e:
        log.error(f'Error running the training: {e}', exc_info=True)