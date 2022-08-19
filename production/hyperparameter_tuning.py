"""Processors for the hyperparameter tuning step of the worklow."""
import logging
import os.path as op
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
from optuna.integration.mlflow import MLflowCallback
from ta_lib.hyperparam_tuning.model_tuning import  tuning

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
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


@register_processor("hyper_parmeter", "parameter-searchSpace")
def model_tuning(context, params):
    """Hyperparameter tuning using Optuna"""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds_train = "train/housing/features"
    input_target_ds_train = "train/housing/target"

    # load datasets
    data = load_dataset(context, input_features_ds_train)
    feature = load_dataset(context, input_target_ds_train)

    # split into training, testing datasets
    train_X,test_X,train_y,test_y=train_test_split(data,feature,test_size=0.2,random_state=42)

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

    # testing data
    test_X = get_dataframe(
        features_transformer.transform(test_X),
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
    test_X = test_X[curated_columns]
    
    # Model Tuning using Optuna
    def objective(trial):
        clf = tuning(trial)
        clf.fit(train_X, train_y.values.ravel())
        y_pred = clf.predict(test_X)
        error = mean_squared_error(test_y, y_pred)
        return error
    mlflc = MLflowCallback(tracking_uri='mlruns',metric_name='mean_squared_error')
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),direction="minimize")  # Create a new study.
    study.optimize(objective,n_trials=500,callbacks=[mlflc])  # Invoke optimization of the objective function.
    trial = study.best_trial
    print("Number of finished trials: {}".format(len(study.trials)))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    # saving results
    filehandler = open(op.abspath(op.join(artifacts_folder, "best_params")),'wb')
    pickle.dump(trial.params, filehandler)