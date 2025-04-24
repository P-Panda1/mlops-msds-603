from metaflow import FlowSpec, step, Parameter, Flow, JSONType, conda_base


@conda_base(python="3.10", libraries={"scikit-learn": "1.4.0", "pandas": "2.1.4"})
class ScoringFlowGCP(FlowSpec):

    @step
    def start(self):
        # Get the latest SUCCESSFUL training run
        training_flow = Flow('TrainingFlowGCP')
        run = training_flow.latest_successful_run

        if not run:
            raise Exception("❌ No successful training runs found!")

        # Access data from the END step (not evaluate_models)
        self.model = run.data.best_model
        self.X_test = run.data.train_data['X_test']
        self.y_test = run.data.train_data['y_test']
        self.model_name = run.data.model_name

        self.next(self.score)

    @step
    def score(self):
        preds = self.model.predict(self.X_test)
        acc = (preds == self.y_test).mean()
        print(f"🤖 Model Used: {self.model_name}")
        print(f"📊 Test Accuracy: {acc:.2f}")
        self.next(self.end)

    @step
    def end(self):
        print("✅ Scoring flow complete!")


if __name__ == '__main__':
    ScoringFlowGCP()
