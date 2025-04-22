from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from dataprocessing import *


class TrainingFlow(FlowSpec):

    seed = Parameter("seed", default=42)
    C = Parameter("C", default=1.0)

    @step
    def start(self):
        import numpy as np
        X_train, X_test, y_train, y_test = load_and_split_data()
        X_train_scaled, X_test_scaled, scaler = scale_data(
            X_train, X_test)

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        self.next(self.train_model)

    @step
    def train_model(self):
        model = LogisticRegression(
            C=self.C, random_state=self.seed, max_iter=1000)
        model.fit(self.X_train, self.y_train)
        self.model = model
        self.next(self.evaluate)

    @step
    def evaluate(self):
        acc = self.model.score(self.X_test, self.y_test)
        self.accuracy = acc
        print(f"Validation accuracy: {acc}")
        self.next(self.register_model)

    @step
    def register_model(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("breast_cancer_experiment")

        with mlflow.start_run():
            mlflow.log_param("C", self.C)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.sklearn.log_model(
                self.model, "model", registered_model_name="BreastCancerClassifier")

        self.next(self.end)

    @step
    def end(self):
        print("Training flow completed.")


if __name__ == '__main__':
    TrainingFlow()
