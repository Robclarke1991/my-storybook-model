from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self):
        """This will run once when the model boots."""
        print("Hello from setup! The simple model has booted successfully.")

    def predict(
        self,
        text: str = Input(description="Text to echo back")
    ) -> str:
        """This will run for each prediction."""
        print(f"Prediction received with text: {text}")
        return f"The simple model received your message: {text}"
