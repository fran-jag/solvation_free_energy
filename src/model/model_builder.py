"""
Model builder service to train and save a ML model.
"""

from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelBuilderService():
    """
    Class to build a ML model
    """

    def __init__(self):
        self.model_path = model_settings.model_path
        self.model_name = model_settings.model_name

    def train_model(self) -> None:
        logger.info(
            'Building model file at ' +
            f'{self.model_path}/{self.model_name}'
        )

    build_model()
