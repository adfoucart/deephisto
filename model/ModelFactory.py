from model import PANModel, ShortResidualModel, UNetModel, BaseModel

class UnknownModelException(Exception):
    def __init__(self, message):
        super().__init__(message)

class ModelFactory:
    """Retrieve model classes from string name"""

    models = {
        'pan': PANModel,
        'shortres': ShortResidualModel,
        'unet': UNetModel
    }

    @classmethod
    def get_model(cls, model_name: str) -> BaseModel:
        """Get model from string name"""
        if model_name not in cls.models:
            raise UnknownModelException(f"Unknown model: {model_name}.")

        return cls.models[model_name]