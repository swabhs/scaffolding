# pylint: disable=no-self-use,invalid-name
import subprocess
import os

from flaky import flaky
import pytest
import numpy

from allennlp.common.testing import ModelTestCase
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.models import Model
from allennlp.nn.util import get_lengths_from_binary_sequence_mask


class SemiCrfSemanticRoleLabelerTest(ModelTestCase):
    def setUp(self):
        super(SemiCrfSemanticRoleLabelerTest, self).setUp()
        self.set_up_model(
            'tests/fixtures/crf_srl/experiment.json', 'tests/fixtures/conll_2012')

    def test_crf_srl_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        class_probs = output_dict['class_probabilities'][0].data.numpy()
        numpy.testing.assert_almost_equal(numpy.sum(class_probs, -1),
                                          numpy.ones(class_probs.shape[0]))

    def test_decode_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.decode(output_dict)
        lengths = get_lengths_from_binary_sequence_mask(
            decode_output_dict["mask"]).data.tolist()
        # Hard to check anything concrete which we haven't checked in the above
        # test, so we'll just check that the tags are equal to the lengths
        # of the individual instances, rather than the max length.
        for prediction, length in zip(decode_output_dict["tags"], lengths):
            assert len(prediction) == length

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the phrase layer wrong - it should be 150 to match
        # the embedding + binary feature dimensions.
        params["model"]["stacked_encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))
