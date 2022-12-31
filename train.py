
import os
import json
import argparse
from typing import Optional, List
import numpy as np
import pandas as pd
from flaml import AutoML as FlamlAutoML

import logging
logger = logging.getLogger()

import warnings
warnings.filterwarnings('ignore')

TRAIN_LOG_FILE = 'experiment_info.log'
MODEL_PARAMS_FILE = 'model_params.txt'


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TabularDataset():
    """Defines a Dataset of columns stored in Parquet, ORC, or CSV format, or from hive table."""

    def __init__(self,
                 path: str,
                 feature_cols: List[str] = None,
                 label_cols: List[str] = None):

        logger.info(f"data loading ...")

        self.pd_frame = pd.read_csv(path)

        self.feature_cols = feature_cols
        self.label_cols = label_cols

    def to_ndarray(self):
        """
        Get features and labels dataframe as training input.

        Returns:
            Dataframe of features and labels
        """
        assert self.feature_cols is not None
        features = self.pd_frame[self.feature_cols].values
        if self.label_cols is not None:
            labels = self.pd_frame[self.label_cols].values
            return features, labels
        else:
            return features


class AutoClassifier():
    def __init__(self,
                 args,
                 train_dataset = None,
                 val_dataset = None):
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        assert self.args.output_dir is not None, "'output_dir' must given."

        self.train_settings = self.build_train_settings(self.args)
        self.auto_trainer = FlamlAutoML()

        self.larger_better_loss_list = ['accuracy', 'roc_auc', 'roc_auc_ovr',
                                        'roc_auc_ovo', 'f1', 'ap', 'micro_f1', 'macro_f1']

    @staticmethod
    def build_train_settings(args):
        if args.log_file_name is None:
            os.makedirs(args.output_dir, exist_ok=True)
            args["log_file_name"] = os.path.join(args.output_dir, TRAIN_LOG_FILE)

        train_settings = {}
        params_list = ['task',
                       'seed',
                       'time_budget',
                       'metric',
                       'estimator_list',
                       'log_file_name',
                       'estimator_kwargs']

        for param_key in params_list:
            param = args.get(param_key)
            if param is not None:
                if param_key == 'estimator_kwargs':
                    if isinstance(param, dict):
                        train_settings.update(param)
                else:
                    train_settings[param_key] = param
        return train_settings

    def train(self):
        logger.info('Train model by auto trainer')
        (X_train, y_train) = self.train_dataset.to_ndarray()
        # logger.info('X_train.dtypes: {}'.format(X_train.dtype))
        # logger.info('y_train.dtypes: {}'.format(y_train.dtype))

        if self.val_dataset is not None:
            (X_val, y_val) = self.val_dataset.to_ndarray()
            self.auto_trainer.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **self.train_settings)
        else:
            self.auto_trainer.fit(X_train=X_train, y_train=y_train, **self.train_settings)

        logger.info('Retrieve best config and best learner')
        logger.info('Best ML leaner: {}'.format(self.auto_trainer.best_estimator))
        logger.info('Best hyper parameter config:{}'.format(self.auto_trainer.best_config))
        if self.auto_trainer.best_config_train_time:
            logger.info('Training duration of best run: {0:.4g} s'.format(self.auto_trainer.best_config_train_time))

        best_eval = self.auto_trainer.best_loss
        if self.train_settings["metric"] in self.larger_better_loss_list:
            best_eval = 1 - best_eval
        logger.info('Best {0} : {1:.4g}'.format(self.train_settings["metric"], best_eval))

    def eval(self):
        (X_val, y_val) = self.val_dataset.to_ndarray()

        logger.info("Evaluation on validation dataset")
        y_pred = self.auto_trainer.predict(X_val)
        y_pred_proba = self.auto_trainer.predict_proba(X_val)[:, 1]
        from flaml.ml import sklearn_metric_loss_score
        logger.info('Accuracy = {}'.format(1 - sklearn_metric_loss_score('accuracy', y_pred, y_val)))

        n_class = len(np.unique(y_val))
        if n_class == 2:
            logger.info('roc_auc = {}'.format(1 - sklearn_metric_loss_score('roc_auc', y_pred_proba, y_val)))
            logger.info('f1 = {}'.format(1 - sklearn_metric_loss_score('f1', y_pred, y_val)))
        else:
            logger.info('micro_f1 = {}'.format(1 - sklearn_metric_loss_score('micro_f1', y_pred, y_val)))
            logger.info('macro_f1 = {}'.format(1 - sklearn_metric_loss_score('macro_f1', y_pred, y_val)))

    def save_model(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = self.args.output_dir

        params_file = os.path.join(output_dir, MODEL_PARAMS_FILE)
        logger.info("Save model params to file: {}".format(params_file))
        self.auto_trainer.save_best_config(params_file)

        model_file = os.path.join(output_dir, self.auto_trainer.best_estimator + '.model')
        logger.info("Save model to: {}".format(model_file))
        if self.auto_trainer.best_estimator == 'lgbm':
            self.auto_trainer.model.estimator.booster_.save_model(model_file)
        elif self.auto_trainer.best_estimator == 'xgboost':
            self.auto_trainer.model.estimator.save_model(model_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AutoML Classifier')

    # data args
    parser.add_argument('--train_input', type=str, help='train input')
    parser.add_argument('--val_input', type=str, help='[optional] validation input')
    parser.add_argument('--output_dir', type=str, help='output directory for model to save')

    parser.add_argument('--data_resample_size', type=int, help='data resample size')
    parser.add_argument('--feature_cols', type=str, help='[optional] features column list')
    parser.add_argument('--label_cols', type=str, default='label', help='data label, default:"label"')

    # trainer args
    parser.add_argument('--time_budget', type=int, default=60,
                        help='constrains the wall-clock time (seconds) used by the AutoML process. default: 60')
    parser.add_argument('--seed', type=int, default=0, help='seed, default:0')
    parser.add_argument('--estimator_list', type=str, default='["lgbm"]',
                        help=""" estimator list:
                             'lgbm': LGBMEstimator for task "classification", "regression". 
                             'xgboost': XGBoostSkLearnEstimator for task "classification", "regression".
                             """)
    parser.add_argument('--estimator_kwargs', type=str,
                        help="""[optional] estimator_kwargs: params dict for the given estimator""")
    parser.add_argument('--metric', type=str, default='log_loss',
                        choices=['accuracy', 'log_loss', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'f1', 'micro_f1', 'macro_f1'],
                        help='optimization metric')

    args = parser.parse_args()
    args = dotdict(vars(args))
    args = prepare_args(args)

    return args


def prepare_args(args):
    if args.train_input is None:
        raise FileExistsError("Error! Train dataset must be given.")

    if args.feature_cols:
        args.feature_cols = eval(args.feature_cols)
    if args.label_cols:
        args.label_cols = eval(args.label_cols)

    if args.estimator_list:
        args.estimator_list = eval(args.estimator_list)
    if args.estimator_kwargs:
        args.estimator_kwargs = eval(args.estimator_kwargs)
    else:
        args.estimator_kwargs = dict()

    args.estimator_kwargs = dotdict(args.estimator_kwargs)
    if args.estimator_kwargs.n_jobs is None:
        args.estimator_kwargs["n_jobs"] = -1
    if args.estimator_kwargs.n_concurrent_trials is None:
        args.estimator_kwargs["n_concurrent_trials"] = 1

    if args.node_resources:
        args.node_resources = eval(args.node_resources)
    else:
        args.node_resources = dict()
    args.node_resources = dotdict(args.node_resources)

    if args.search_space:
        args.search_space = json.loads(args.search_space)

    args.task = 'classification'

    return args


def train_classifier(args):

    # dataset
    train_dataset = TabularDataset(path=args.train_input,
                                   feature_cols=args.feature_cols,
                                   label_cols=args.label_cols)

    if args.data_resample_size:
        train_dataset.pd_frame = train_dataset.pd_frame.sample(n=args.data_resample_size, random_state=1, replace=True)

    logger.info(f"resampleed data shape:{train_dataset.pd_frame.shape}")

    val_dataset = None
    if args.val_input is not None:
        val_dataset = TabularDataset(path=args.val_input,
                                     feature_cols=args.feature_cols,
                                     label_cols=args.label_cols)

    # train process
    classifier = AutoClassifier(args, train_dataset, val_dataset)
    classifier.train()

    # evaluation
    if val_dataset:
        classifier.eval()

    # save model
    classifier.save_model()


def main():
    # parse args
    args = parse_args()
    train_classifier(args)


if __name__ == '__main__':
    main()

