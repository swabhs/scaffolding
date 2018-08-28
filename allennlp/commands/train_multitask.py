"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python -m allennlp.run train --help
   usage: run [command] train [-h] -s SERIALIZATION_DIR param_path

   Train the specified model on the specified dataset.

   positional arguments:
   param_path            path to parameter file describing the model to be trained

   optional arguments:
    -h, --help            show this help message and exit
    -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the model and its logs
"""
from typing import Dict
import argparse
import json
import logging
import os
import random
import sys
from copy import deepcopy

from allennlp.commands.evaluate import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.common.util import prepare_environment
from allennlp.data import Dataset, Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model
from allennlp.models.model import Model
from allennlp.training.multitask_trainer import MultiTaskTrainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class TrainMultitask(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(
            name, description=description, help='Train a model')
        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        # This is necessary to preserve backward compatibility
        serialization = subparser.add_mutually_exclusive_group(required=True)
        serialization.add_argument('-s', '--serialization-dir',
                                   type=str,
                                   help='directory in which to save the model and its logs')
        serialization.add_argument('--serialization_dir',
                                   type=str,
                                   help=argparse.SUPPRESS)

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path, args.serialization_dir)


def train_model_from_file(parameter_filename: str, serialization_dir: str) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path: str, required.
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename)
    return train_model(params, serialization_dir)


def train_model(params: Params, serialization_dir: str) -> Model:
    """
    This function can be used as an entry point to running models in AllenNLP
    directly from a JSON specification using a :class:`Driver`. Note that if
    you care about reproducibility, you should avoid running code using Pytorch
    or numpy which affect the reproducibility of your experiment before you
    import and use this function, these libraries rely on random seeds which
    can be set in this function via a JSON specification file. Note that this
    function performs training and will also evaluate the trained model on
    development and test sets if provided in the parameter json.

    Parameters
    ----------
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    prepare_environment(params)

    os.makedirs(serialization_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(
        serialization_dir, "stdout.log"), sys.stdout)  # type: ignore
    sys.stderr = TeeLogger(os.path.join(
        serialization_dir, "stderr.log"), sys.stderr)  # type: ignore
    handler = logging.FileHandler(os.path.join(
        serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    # Now we begin assembling the required parts for the Trainer.

    # 1. Primary training data.
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    # 2. Auxillary training data.
    dataset_reader_aux = DatasetReader.from_params(
        params.pop('dataset_reader_aux'))
    train_data_path_aux = params.pop('train_data_path_aux')
    logger.info("Reading auxilliary training data from %s",
                train_data_path_aux)
    train_data_aux = dataset_reader_aux.read(train_data_path_aux)

    # If only using a fraction of the auxiliary data.
    aux_sample_fraction = params.pop("aux_sample_fraction", 1.0)
    if aux_sample_fraction < 1.0:
        sample_size = int(aux_sample_fraction * len(train_data_aux.instances))
        train_data_aux = Dataset(random.sample(train_data_aux.instances, sample_size))

    # Balance the two datasets by inflating the size of the smaller dataset to the size of the larger dataset.
    train_size = len(train_data.instances)
    aux_train_size = len(train_data_aux.instances)
    mixing_ratio = params.pop("mixing_ratio")
    # mixing_ratio = float(train_size)/aux_train_size

    if train_size > aux_train_size:  # case for PB scaffold.
        difference = train_size - aux_train_size
        aux_sample = [random.choice(train_data_aux.instances) for _ in range(difference)]
        train_data_aux = Dataset(train_data_aux.instances + aux_sample)
        logger.info("Inflating auxiliary train data from {} to {} samples".format(
            aux_train_size, len(train_data_aux.instances)))
    # else: # case for FN scaffold.
    #     difference = aux_train_size - train_size
    #     train_sample = [random.choice(train_data.instances) for _ in range(difference)]
    #     train_data = Dataset(train_data.instances + train_sample)
    #     logger.info("Inflating train data from {} to {} samples".format(
    #         train_size, len(train_data.instances)))

    all_datasets: Dict[str, Dataset] = {"train": train_data}
    all_datasets_aux: Dict[str, Dataset] = {"train_aux": train_data_aux}

    # 3. Primary validation data.
    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)
        all_datasets["validation"] = validation_data
    else:
        validation_data = None

    # 4. Auxillary validation data.
    validation_data_path_aux = params.pop('validation_data_path_aux', None)
    if validation_data_path_aux is not None:
        logger.info("Reading auxilliary validation data from %s",
                    validation_data_path_aux)
        validation_data_aux = dataset_reader_aux.read(validation_data_path_aux)
        all_datasets_aux["validation_aux"] = validation_data_aux
    else:
        validation_data_aux = None

    # 5. Primary test data
    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = dataset_reader.read(test_data_path)
        all_datasets["test"] = test_data
    else:
        test_data = None

    # 6. Auxillary test data
    test_data_path_aux = params.pop("test_data_path_aux", None)
    if test_data_path_aux is not None:
        logger.info("Reading auxillary test data from %s", test_data_path_aux)
        test_data_aux = dataset_reader_aux.read(test_data_path_aux)
        all_datasets_aux["test_aux"] = test_data_aux
    else:
        test_data_aux = None

    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))
    datasets_for_vocab_creation_aux = set(params.pop("auxillary_datasets_for_vocab_creation", all_datasets_aux))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(
                f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("Creating a vocabulary using %s data. Auxillary also included.",
                ", ".join(datasets_for_vocab_creation))
    dataset_primary = Dataset([instance for key, dataset in all_datasets.items()
                               for instance in dataset.instances
                               if key in datasets_for_vocab_creation])
    dataset_aux = Dataset([instance for key, dataset in all_datasets_aux.items()
                           for instance in dataset.instances
                           if key in datasets_for_vocab_creation_aux])
    vocab = Vocabulary.from_params(params.pop("vocabulary", {}),
                                   dataset_primary,
                                   dataset_aux=dataset_aux)
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    model = Model.from_params(vocab, params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator_aux = DataIterator.from_params(params.pop("iterator_aux"))

    train_data.index_instances(vocab)
    train_data_aux.index_instances(vocab)
    if validation_data:
        validation_data.index_instances(vocab)
    if validation_data_aux:
        validation_data_aux.index_instances(vocab)

    cutoff_epoch = params.pop("cutoff_epoch", -1)

    trainer_params = params.pop("trainer")
    trainer = MultiTaskTrainer.from_params(model=model,
                                           serialization_dir=serialization_dir,
                                           iterator=iterator,
                                           iterator_aux=iterator_aux,
                                           train_dataset=train_data,
                                           train_dataset_aux=train_data_aux,
                                           mixing_ratio=mixing_ratio,
                                           cutoff_epoch=cutoff_epoch,
                                           validation_dataset=validation_data,
                                           validation_dataset_aux=validation_data_aux,
                                           params=trainer_params,
                                           files_to_archive=params.files_to_archive)

    evaluate_on_test = params.pop("evaluate_on_test", False)
    params.assert_empty('base train command')
    trainer.train()

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    if test_data and evaluate_on_test:
        test_data.index_instances(vocab)
        evaluate(model, test_data, iterator,
                 cuda_device=trainer._cuda_device)  # pylint: disable=protected-access

    elif test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    if test_data_aux and evaluate_on_test:
        test_data_aux.index_instances(vocab)
        evaluate(model, test_data_aux, iterator_aux,
                 cuda_device=trainer._cuda_device)  # pylint: disable=protected-access

    elif test_data_aux:
        logger.info("To evaluate on the auxillary test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    return model
