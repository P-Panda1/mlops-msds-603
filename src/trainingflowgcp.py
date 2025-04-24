from metaflow import FlowSpec, step, Parameter, kubernetes, catch, retry, timeout, conda_base
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataprocessing import *


@conda_base(python="3.10", libraries={"scikit-learn": "1.4.0", "pandas": "2.1.4"})
class TrainingFlowGCP(FlowSpec):

    seed = Parameter("seed", default=42)
    C = Parameter("C", default=1.0)

    @catch  # Catch any exceptions and continue (applied above @step)
    @step
    def start(self):
        X_train, X_test, y_train, y_test = load_and_split_data()
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

        self.train_data = {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test
        }
        self.next(self.train_models)

    @step
    def train_models(self):
        X_train = self.train_data["X_train"]
        y_train = self.train_data["y_train"]

        # Train Logistic Regression
        lr_model = LogisticRegression(
            C=self.C, random_state=self.seed, max_iter=1000)
        lr_model.fit(X_train, y_train)

        # Train Random Forest
        rf_model = RandomForestClassifier(random_state=self.seed)
        rf_model.fit(X_train, y_train)

        self.lr_model = lr_model
        self.rf_model = rf_model
        self.train_data = self.train_data  # Carry forward test set
        self.next(self.evaluate_models)

    @step
    def evaluate_models(self):
        X_test = self.train_data["X_test"]
        y_test = self.train_data["y_test"]

        acc_lr = self.lr_model.score(X_test, y_test)
        acc_rf = self.rf_model.score(X_test, y_test)

        print(f"📊 Logistic Regression Accuracy: {acc_lr}")
        print(f"🌲 Random Forest Accuracy: {acc_rf}")

        if acc_lr >= acc_rf:
            self.best_model = self.lr_model
            self.model_name = "LogisticRegression"
            self.best_acc = acc_lr
        else:
            self.best_model = self.rf_model
            self.model_name = "RandomForest"
            self.best_acc = acc_rf

        self.train_data = self.train_data
        self.next(self.end)

    @step
    def end(self):
        print(
            f"✅ Training complete. Best model: {self.model_name} with accuracy {self.best_acc}")


if __name__ == '__main__':
    TrainingFlowGCP()
