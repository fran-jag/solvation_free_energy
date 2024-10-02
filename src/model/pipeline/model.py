"""
Module to create, train or load a model and saving.
"""

import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from loguru import logger

from config import model_settings
from model.pipeline.preparation import get_descriptor_df


def build_model():
    logger.info('Starting model building pipeline...')
    df = get_descriptor_df()
    # 2. Extract features X and target y
    x, y = _get_x_y(df)
    # 3. Train-test split
    x_train, x_test, y_train, y_test = _split_train_test(x, y)
    # 4. Train and tune model
    logger.info('Training and tunning model...')
    model = train_model(x_train, y_train)
    # 5. Save tunned model
    _save_model(model)
    logger.info('Model saved.')


def _get_x_y(data, col_x=None, col_y='label'):

    if col_x is None:
        col_x = data.columns[:-1]

    return data[col_x].values, data[col_y].values


def _split_train_test(x, y, frac=0.2):

    return train_test_split(x, y, test_size=frac)


def train_model(x_train, y_train):

    grid_space = {'n_estimators': [100, 500],
                  'max_features': ['sqrt', 'log2', 0.01, 0.1, 0.5],
                  "max_depth": [3, 6, 9, 12]}

    grid = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=grid_space,
        cv=5,
        scoring='r2',
    )
    model_grid = grid.fit(x_train, y_train)
    return model_grid.best_estimator_


def _evaluate_model(model, x_test, y_test):
    score = model.score(x_test, y_test)
    return score


def _save_model(model):
    model_file = f'{model_settings.model_path}/{model_settings.model_name}'
    with open(model_file,
              'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    build_model()
