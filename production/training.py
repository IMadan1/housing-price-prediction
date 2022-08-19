"""Processors for the model training step of the worklow."""
import imp
import logging
import os.path as op
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
)
from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    # transform the training data
    train_X = get_dataframe(
        features_transformer.fit_transform(train_X, train_y),
        [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "rooms_per_household",
            "population_per_household",
            "bedrooms_per_room",
            "ocean_proximity_<1H OCEAN",
            "ocean_proximity_INLAND",
            "ocean_proximity_ISLAND",
            "ocean_proximity_NEAR BAY",
            "ocean_proximity_NEAR OCEAN",
        ],
    )
    train_X = train_X[curated_columns]
    
    # create training pipeline
    param=pd.read_pickle(op.join(artifacts_folder, "best_params"))
    regressor = param['regressor']
    param.pop('regressor')
    if regressor=="Linear":
        model = Pipeline(
            [("linear", SKLStatsmodelOLS(**param))]
        )
    elif regressor=='DecisionTree':
        model = Pipeline(
            [("decision_tree", DecisionTreeRegressor(**param))]
        )
    else:
        model = Pipeline(
            [("random_forest", RandomForestRegressor(**param))]
        )

    # fit the training pipeline
    model.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(model, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib")))