from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from ta_lib.regression.api import SKLStatsmodelOLS
import optuna

def tuning(trial):
    """Hyper-parameter tuning of models.

    Parameters
    ----------
    trial: A single call of the objective function
    
    Returns
    -------
    object
        Initialized Model object of optimized model and parameters
    """
    regressor_name = trial.suggest_categorical('regressor', ['Linear','DecisionTree','RandomForest'])
    if regressor_name == 'Linear':
        model=SKLStatsmodelOLS()
    else:
        params = {
            'ccp_alpha':trial.suggest_float('ccp_alpha',0.1,0.2 ,log=True),
            'min_samples_split':trial.suggest_int('min_samples_split', 2,8),
            'max_features':trial.suggest_int('max_features', 2,10),
            'max_depth':trial.suggest_int('max_depth', 3,20)}
        if regressor_name == 'DecisionTree':
            model = DecisionTreeRegressor(**params)
        else:
            params['n_estimators']=trial.suggest_int('n_estimators', 2, 30)
            model = RandomForestRegressor(**params)
    return model