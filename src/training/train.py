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
import torch.nn as nn
import logging as log
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import TypedDict

from torch.nn import Module
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)
from mlflow import pytorch as mlflow_pytorch
from mlflow.models.signature import infer_signature

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.processing.main import CreditDataProcessor
from src.training.model import CreditScoringModel

Metrics_history = TypedDict('Metrics_history', {
    "train_loss": List[float],
    "val_loss": list[float],
    "train_acc": list[float],
    "val_acc": list[float],
    "train_auc": list[float],
    "val_auc": list[float]
})

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

        self.history: Metrics_history = {
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

    def _preprocess_data(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[torch.Tensor, ...]:
        """
        Fits preprocessor on training data and transforms both sets
        """

        log.info("Processing data...")
        preprocessor = self.data_processor.fit_preprocessor(df_train)

        x_train_processed, y_train_processed = self.data_processor.process_data(df_train, preprocessor)
        x_val_processed, y_val_processed = self.data_processor.process_data(df_val, preprocessor)

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
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, filename: str):
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
    
    def _plot_roc_pr(self, y_true: np.ndarray, y_prob: np.ndarray, roc_file: str, pr_file: str):
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
    
    def _run_training_loop(self, model: CreditScoringModel, criterion: Module, optimizer: optim.Adam | optim.AdamW | optim.SGD, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = None):
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

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            train_metrics = self._evaluate_split(model, x_train, y_train, criterion)
            val_metrics = self._evaluate_split(model, x_val, y_val, criterion)

            if scheduler:
                scheduler.step(val_metrics['loss'])

            # Save History
            self._save_metrics_history(train_metrics, val_metrics)

            # logs -> console
            current_lr = optimizer.param_groups[0]["lr"]
            log.info(
                f"✔ Epoch [{epoch}/{self.epochs}]"
                f"✔ TrainLoss: {train_metrics['loss']:.4f} |  ValLoss: {val_metrics['loss']:4f}"
                f"✔ TrainAcc: {train_metrics['accuracy']:.4f} | ValAcc: {val_metrics['accuracy']:.4f}"
                f"✔ TrainAUC: {train_metrics['roc_auc']:.4f} | ValAUC: {val_metrics['roc_auc']:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "val_precision": val_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_recall": val_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "train_roc_auc": train_metrics["roc_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "lr": current_lr
            }, step=epoch)

            if val_metrics["loss"] < best_val_loss - self.early_stopping_delta:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                path_model = f"models/{self.model_name}"
                torch.save(model.state_dict(), path_model)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    log.info(f"✘ Early stopping activated in epoch {epoch}")
                    break
            
        log.info("---- Training finished ----")
        return epochs_run
        
    def _generate_and_log_performance_report(
            self,
            model: CreditScoringModel,
            final_metrics: Dict[str, float],
            num_features: int,
            epochs_run: int,
            run_name: str
    ):
        """
        Generates a YAML report, save it locally, and logs it to MLflow.
        """

        log.info("---- Generating performance report ----")
        model_info = model.get_model_info()

        report_data = {
            "benchmark_id": self.params.get("project_info", {}).get("benchmark_id", "N/A"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_architecture": {
                "model_type": model_info["model_type"],
                "input_features": num_features,
                "hidden_layers": model_info["architecture"]["hidden_layers"],
                "use_batch_norm": model_info["use_batch_norm"],
                "activation_fn": model_info["activation_fn"],
                "dropout_rate": model_info["dropout_rate"],
                "output_layer_neurons": model_info["architecture"]["output_layer"],
                "total_parameters": model_info["total_parameters"],
            },
            "training_configuration": {
                "optimizer": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "loss_function": "BCEWithLogitsLoss",
                "use_pos_weight": self.use_pos_weight,
                "epochs_run": epochs_run,
                "batch_size": self.batch_size,
            },
            "final_validation_metrics": {k: round(v, 4) for k, v in final_metrics.items() if not math.isnan(v)}
        }

        # Save locally
        report_filename = f"{run_name}_performance_report.yml"
        report_path = self.local_artifacts_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report_data, f, indent=2, sort_keys=False)

        log.info(f"Performance report saved locally to: {report_path}")

        mlflow.log_artifact(str(report_path), artifact_path="reports")
        log.info("Perfomance report logged to MLflow artifacts.")

    def _log_basic_params(self, num_features: int):
        mlflow.log_params({
            "test_size": self.test_size, 
            "random_state": self.random_state,
            "num_features": num_features, 
            "epochs": self.epochs, 
            "batch_size": self.batch_size,
            "hidden_layers": str(self.hidden_layers),
            "use_batch_norm": self.use_batch_norm,
            "activation_fn": self.activation_fn,
            "dropout_rate": self.dropout_rate,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "use_pos_weight": self.use_pos_weight,
            "scheduler_patience": self.scheduler_patience,
            "early_stopping_patience": self.early_stopping_patience
        })

        tags = self.params.get("mlflow_config", {}).get("mlflow_tags", [])
        if tags:
            for i, t in enumerate(tags):
                mlflow.set_tag(f"tag_{i}", t)

    def _log_model_with_signature(self, model: nn.Module, x_example: torch.Tensor):
        model_cpu = model.to("cpu").eval()

        with torch.no_grad():
            y_example = model_cpu(x_example).numpy()

        # signature e input example
        signature = infer_signature(x_example.numpy(), y_example)
        input_example = x_example.numpy()

        pip_requirements = [
            # "-f https://download.pytorch.org/whl/cu126",
            # "torch==2.7.1+cu126",
            "torch==2.7.1",  # fallback genérico para PyPI
            "scikit-learn",
            "pandas",
            "numpy",
        ]

        try:
            mlflow_pytorch.log_model(
                model_cpu,
                name=self.artifact_name_or_path,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements
            )
        except TypeError:
            mlflow_pytorch.log_model(
                model_cpu,
                artifact_path=self.artifact_name_or_path,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements
            )        

    def _log_plots_and_reports(self, y_true_val: np.ndarray, y_prob_val: np.ndarray):
        epochs = list(range(1, len(self.history["train_loss"]) + 1))

        loss_png = self._plot_and_save(epochs, self.history["train_loss"], self.history["val_loss"],
                                    "Training vs Validation Loss", "Loss", "loss_train_val.png")
        acc_png = self._plot_and_save(epochs, self.history["train_acc"], self.history["val_acc"],
                                    "Training vs Validation Accuracy", "Accuracy", "acc_train_val.png")

        roc_png, pr_png = self._plot_roc_pr(y_true_val, y_prob_val, roc_file="roc_val.png", pr_file="pr_val.png")
        y_pred_val = (y_prob_val >= 0.5).astype(int)

        cm_png = self._plot_confusion_matrix(y_true_val, y_pred_val, filename="confusion_matrix_val.png")

        cls_report = classification_report(y_true_val, y_pred_val, target_names=["bad(0)", "good(1)"])
        report_path = self.local_artifacts_dir / "classification_report_val.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(str(cls_report))

        mlflow.log_artifact(str(loss_png), artifact_path="plots")
        mlflow.log_artifact(str(acc_png), artifact_path="plots")

        if roc_png:
            mlflow.log_artifact(str(roc_png), artifact_path="plots")

        if pr_png:
            mlflow.log_artifact(str(pr_png), artifact_path="plots")

        mlflow.log_artifact(str(cm_png), artifact_path="plots")
        mlflow.log_artifact(str(report_path), artifact_path="reports")

    def _setup_loss_function(self, y_train: torch.Tensor) -> nn.Module:
        """Configures the loss function based on YAML parameters."""
        if self.use_pos_weight:
            y_train_cpu = y_train.detach().cpu().numpy().reshape(-1)
            pos = float(np.sum(y_train_cpu == 1))
            neg = float(np.sum(y_train_cpu == 0))
            
            if pos == 0 or neg == 0:
                log.warning("Una de las clases no está presente en el batch de entrenamiento. No se usará pos_weight.")
                return nn.BCEWithLogitsLoss()
            
            pos_weight_value = neg / pos
            pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=self.device)
            log.info(f"✔ using weighted BCE loss with pos_weight={pos_weight_value:.4f}")
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            log.info("✔ using standard BCE loss.")
            return nn.BCEWithLogitsLoss()

    def _save_metrics_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        self.history['train_loss'].append(train_metrics["loss"])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['train_auc'].append(val_metrics['roc_auc'])
        self.history['val_auc'].append(val_metrics['roc_auc'])

    def train(self):
        """Main method to orchestrate the model training pipeline."""
        log.info(f"✔ hardware used: {self.device}")
        
        mlflow.set_experiment(self.mlflow_project_name)
        run_name_prefix = self.params['mlflow_config'].get('mlflow_run_name_prefix', 'credit_scoring_run')

        with mlflow.start_run(run_name=f"{run_name_prefix}"):
            log.info("--- Init Training ---")
            
            # 1. Load and split data
            df_train, df_val = self._load_and_split_data()
            
            # 2. Preprocess data
            x_train, y_train, x_val, y_val = self._preprocess_data(df_train, df_val)
            num_features = x_train.shape[1]
            
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            x_val, y_val = x_val.to(self.device), y_val.to(self.device)
            
            # 3. Configure model, optimizer, and loss function
            log.info(f"✔ initializing model with config: {self.hidden_layers}")
            log.info(f"✔ initializing model with {num_features} input features.")
            model = CreditScoringModel(
                num_features=num_features,
                hidden_layers=self.hidden_layers,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
                activation_fn=self.activation_fn).to(self.device)
            
            # loss function
            criterion = self._setup_loss_function(y_train)
            
            # optimizer
            if self.optimizer_name.lower() == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            elif self.optimizer_name.lower() == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            elif self.optimizer_name.lower() == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            else:
                raise ValueError(f"Optimizer {self.optimizer_name} not supported.")
            log.info(f"✔ using optimizer: {self.optimizer_name} with lr={self.learning_rate}")
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.scheduler_factor, patience=self.scheduler_patience)
            
            # Log parameters to MLflow
            self._log_basic_params(num_features=num_features)
            
            # 4. Run training loop
            epochs_run = self._run_training_loop(model, criterion, optimizer, x_train, y_train, x_val, y_val, scheduler)
            
            # 5. Load best model and log artifacts
            path_model = f"models/{self.model_name}"
            log.info(f"✔ Loading best model from {path_model} and logging artifacts.")
            model.load_state_dict(torch.load(path_model, map_location=self.device))
            model.eval()
            
            with torch.no_grad():
                logits_val = model(x_val)
                prob_val = torch.sigmoid(logits_val).detach().cpu().numpy().reshape(-1)
                y_val_np = y_val.detach().cpu().numpy().reshape(-1)
                
            final_metrics = self._compute_metrics(y_val_np, prob_val, threshold=0.5)
            mlflow.log_metrics({f"final_val_{k}": v for k, v in final_metrics.items() if not math.isnan(v)})

            # 6. plots & reports
            self._log_plots_and_reports(y_val_np, prob_val)
            
            # 7. log model
            self._generate_and_log_performance_report(model, final_metrics, num_features, epochs_run, run_name_prefix)
            x_example = x_train[:5].detach().cpu()
            self._log_model_with_signature(model, x_example)
            path_preprocessor = f"models/{self.preprocessor_filename}"
            mlflow.log_artifact(path_preprocessor, artifact_path="preprocessing")
            log.info("✔ Preprocessor and model save in MLflow.")

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
        trainer.train()
    except Exception as e:
        log.error(f'Error running the training: {e}', exc_info=True)