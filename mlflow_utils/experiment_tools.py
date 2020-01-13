import mlflow
import tempfile
import os
from mlflow.tracking import MlflowClient
from enum import Enum, auto

import mlflow.sklearn
import mlflow.xgboost


class Models(Enum):
    SKLEARN = auto()
    XGBOOST = auto()
    OTHER = auto()


class ExperimentLogger:
    def __init__(self, experiment_id: str, model_type: Models.OTHER):

        if not isinstance(model_type, Models):
            raise AttributeError("Invalid model type. Specify using Models class.")

        self.experiment_id = experiment_id
        self.model_type = model_type

    def _get_model_params(self, model: 'Model class instance'):
        """
        Discovers model related parameters for known model types.

        Returns
        -------
        params (dict): Dictionary containing class_name:str, params:dict and model_logger: function
        """

        if self.model_type == Models.SKLEARN:
            return {
                'class_name': model.__class__.__name__,
                'params': model.get_params(),
                'model_logger': mlflow.sklearn.log_model
            }
        elif self.model_type == Models.XGBOOST:
            return {
                'class_name': model.__class__.__name__,
                'params': model.get_params(),
                'model_logger': mlflow.xgboost.log_model
            }
        else:
            return None

    def log_experiment(self, model, features: list, metrics: dict, tag: 'tuple or str' = None,
                       matplotlib_figures: 'list of tuples' = None, additional_params: dict = None,
                       artifact_root="dbfs:/databricks/mlflow") -> None:
        """
        Wrapper to log experiment into MLFlow. Works with Scikit-learn and XGBoost models.

        Parameters
        ----------

        features (list): list of feature names to log. Feature string will be split into multiple strings of 400 s
        ymbols, which is current limitation of MLFlow.

        metrics (dict): dictionary of metrics, for example  {'train_auc':train_auc, 'test_auc':test_auc}

        tag (tuple or str): tag to add to experiment, in format(key, value), or "key"

        matplotlib_figures (list of tuples): list in format
         [(fig1, "filename1.png"), (fig2, "filename2.png")..] where fig1.. are Matplotlib figure objects
        additional_params (dict): additional parameters, if any...

        Examples
        -------
        Log plain experiment:
        > exp = ExperimentLogger('1234', Models.SKLEARN)
        > exp.log_experiment(cls, feature_cols, metrics = {'train_auc':train_auc, 'test_auc':test_auc})

        Log experiment with Matplotlib figure artifacts:
        > exp = ExperimentLogger('1234', Models.SKLEARN)

        > exp.log_experiment(cls, matplotlib_figures = [(fig, "feat_importance.png")],
        metrics = {'train_auc':train_auc, 'test_auc':test_auc}))
        """

        res = self._get_model_params(model)

        with mlflow.start_run(experiment_id=self.experiment_id):
            if res:
                mlflow.log_param("class_name", res['class_name'])
                if not additional_params:  # Log parameters only if params is empty
                    mlflow.log_params(res['params'])
                res['model_logger'](model, artifact_path=f"{artifact_root}/{self.experiment_id}")  # Saving model
            else:
                print("Unknown model type, couldn't discover parameters")

            features_str_list = split_string(",".join(features))

            for i, f_s in enumerate(features_str_list):
                mlflow.log_param("features_{}".format(i), f_s)

            if additional_params:
                mlflow.log_params(additional_params)

            if metrics:
                mlflow.log_metrics(metrics)

            if tag:
                if isinstance(tag, tuple):
                    mlflow.set_tag(*tag)
                elif isinstance(tag, str):
                    mlflow.set_tag(tag, "")
                else:
                    print("Unable to parse tag.")

            # Save matplotlib artifacts if present
            if matplotlib_figures:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for fig, fn in matplotlib_figures:
                        image_path = os.path.join(temp_dir, fn)
                        fig.savefig(image_path)
                        mlflow.log_artifact(image_path)


def show_experiments():
    client = MlflowClient()
    return client.list_experiments()


def split_string(s: str, max_c: int = 400, split_s: str = ",") -> list:
    """ Splits string to contain max_c characters, preserving natural split.

    Parameters
    ----------

    s (str): input string
    max_c(int): maximum string length
    split_s(char): splitting character

    Returns
    -------

    res_list(list): list of strings
    """

    res_list = []
    res_s = ""
    for word in s.split(split_s):
        if len(res_s + word) < max_c:
            res_s += word + ","
        else:
            res_list.append(res_s[:-1])
            res_s = ""
    return res_list


if __name__ == '__main__':
    print(mlflow)
