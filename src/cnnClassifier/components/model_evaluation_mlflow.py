import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt       
import json                           

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        # Basic loss + accuracy
        self.score = self.model.evaluate(self.valid_generator)
        # Predict for precision, recall, f1
        self._compute_extended_metrics()
        self.save_score()

    def _compute_extended_metrics(self):
        y_pred_probs = self.model.predict(self.valid_generator)
        y_true = self.valid_generator.classes
        num_classes = self.valid_generator.num_classes

        if num_classes == 2:
            if y_pred_probs.shape[1] == 1:
                y_pred = (y_pred_probs[:, 0] > 0.5).astype(int)
            else:
                y_pred = np.argmax(y_pred_probs, axis=1)
            average = "binary"
        else:
            y_pred = np.argmax(y_pred_probs, axis=1)
            average = "weighted"

        print(f"y_pred unique: {np.unique(y_pred)}")        
        print(f"y_true unique: {np.unique(y_true)}")        

        self.precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        self.recall    = recall_score(y_true, y_pred, average=average, zero_division=0)
        self.f1        = f1_score(y_true, y_pred, average=average, zero_division=0)

        print(f"Precision: {self.precision}, Recall: {self.recall}, F1: {self.f1}")  


    def save_score(self):
        scores = {
            "loss":      self.score[0],
            "accuracy":  self.score[1],
            "precision": self.precision,
            "recall":    self.recall,
            "f1_score":  self.f1
            }
        save_json(path=Path("scores.json"), data=scores)

    def log_training_curves(self):                      # ✅ new method
        if not Path("training_history.json").exists():
            print("No training history found, skipping curves.")
            return

        with open("training_history.json", "r") as f:
            history = json.load(f)

        epochs = range(1, len(history["loss"]) + 1)

        # Accuracy curve
        fig, ax = plt.subplots()
        ax.plot(epochs, history["accuracy"], label="Train Accuracy")
        ax.plot(epochs, history["val_accuracy"], label="Val Accuracy")
        ax.set_title("Training vs Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.savefig("accuracy_curve.png")
        mlflow.log_artifact("accuracy_curve.png")
        plt.close(fig)

        # Loss curve
        fig, ax = plt.subplots()
        ax.plot(epochs, history["loss"], label="Train Loss")
        ax.plot(epochs, history["val_loss"], label="Val Loss")
        ax.set_title("Training vs Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.savefig("loss_curve.png")
        mlflow.log_artifact("loss_curve.png")
        plt.close(fig)


    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            # 1. Log parameters
            mlflow.log_params(self.config.all_params)

            # 2. Log metrics
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1],
                "precision": self.precision,
                "recall":    self.recall,
                "f1_score":  self.f1
            })

            # 3. training curves                        # ✅ add this
            self.log_training_curves()

            # 4. Save lightweight model
            self.model.save("model.h5")

            # Upload model artifact
            mlflow.log_artifact("model.h5")

            print("MLflow logging completed successfully")