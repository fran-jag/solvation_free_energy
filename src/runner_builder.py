"""
Main application for running the ML model builder serivice.
"""

from loguru import logger

from model.model_builder import ModelBuilderService


@logger.catch
def main():
    """
    Main function
    """
    logger.info('Running ModelBuilderService')
    ml_svc = ModelBuilderService()
    ml_svc.train_model()


if __name__ == '__main__':
    main()
