from sklearn.utils.estimator_checks import check_estimator


def test_existance():
    from ta_lib.data_processing.estimators import CombinedAttributesAdder
    from ta_lib.hyperparam_tuning.api import tuning


def test_estimator_check():
    from ta_lib.data_processing.estimators import CombinedAttributesAdder
    from ta_lib.hyperparam_tuning.api import tuning

    check_estimator(CombinedAttributesAdder())