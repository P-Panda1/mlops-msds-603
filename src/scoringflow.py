from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
from dataprocessing import *


class ScoringFlow(FlowSpec):

    run_id = Parameter('rid',
                       help="MLflow run ID from which to load the trained model",
                       default=None)

    @step
    def start(self):
        print(f"Using MLflow run ID: {self.run_id}")
        self.next(self.check_run_id)

    @step
    def check_run_id(self):
        if not self.run_id:
            print("❌ No run ID provided.")
            self.valid_run_id = False
        else:
            self.valid_run_id = True

        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        # Load hold-out data
        X_train, X_test, y_train, y_test = load_and_split_data()
        _, X_test_scaled, _ = scale_data(X_train, X_test)

        self.X_test = X_test_scaled
        self.y_test = y_test
        self.next(self.load_model)

    @step
    def load_model(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        model_uri = f"runs:/{self.run_id}/model"
        print(f"✅ Loading model from URI: {model_uri}")
        if self.valid_run_id:
            self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        if not self.valid_run_id:
            print("❌ Invalid run ID. Ending the flow.")
        else:
            preds = self.model.predict(self.X_test)
            acc = (preds == self.y_test).mean()
            print(f"📊 Scoring Accuracy: {acc}")
        self.next(self.end)

    @step
    def end(self):
        print("✅ Scoring flow completed.")


if __name__ == '__main__':
    ScoringFlow()
