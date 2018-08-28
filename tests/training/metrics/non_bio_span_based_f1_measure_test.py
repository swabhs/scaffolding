# pylint: disable=no-self-use,invalid-name,protected-access
import torch
import numpy as np

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.training.metrics import NonBioSpanBasedF1Measure, Metric
from allennlp.common.params import Params


class NonBioSpanBasedF1Test(AllenNlpTestCase):

    def setUp(self):
        super(NonBioSpanBasedF1Test, self).setUp()
        vocab = Vocabulary()
        vocab.add_token_to_namespace("*", "tags")
        vocab.add_token_to_namespace("ARG1", "tags")
        vocab.add_token_to_namespace("ARG2", "tags")
        vocab.add_token_to_namespace("O", "tags")
        vocab.add_token_to_namespace("V", "tags")
        self.vocab = vocab

    def test_extract_spans(self):
        # Check that span extraction works without merging.
        max_span_width = 2
        sent_len = 7
        metric = NonBioSpanBasedF1Measure(
            self.vocab, tag_namespace="tags", ignore_classes=["*", "V"])

        tag_matrix = [['*' for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        tag_matrix[1][1] = "ARG1"
        tag_matrix[2][0] = "ARG2"
        tag_matrix[5][0] = "V"
        indices = [[self.vocab.get_token_index(
            x, "tags") for x in tag_list] for tag_list in tag_matrix]
        spans = metric._extract_spans(indices)
        assert spans == {(0, 1, "ARG1"), (2, 2, "ARG2")}

        # Check that span extraction works with merging when nothing to merge.
        spans = metric._extract_spans(indices, merge=True)
        assert spans == {(0, 1, "ARG1"), (2, 2, "ARG2")}

        # Check that completely empty predictions are correct.
        indices = [[self.vocab.get_token_index('*', 'tags') for _ in range(max_span_width)]
                   for _ in range(sent_len)]
        spans = metric._extract_spans(indices)
        assert not spans

        # Check that it works when spans are longer than max_span_width.
        max_span_width = 3
        sent_len = 5
        tag_matrix = [['*' for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        tag_matrix[1][1] = "ARG1"
        tag_matrix[3][1] = "ARG1"
        tag_matrix[4][0] = "ARG1"
        indices = [[self.vocab.get_token_index(
            x, "tags") for x in tag_list] for tag_list in tag_matrix]
        spans = metric._extract_spans(indices)
        assert spans == {(0, 1, "ARG1"), (2, 3, "ARG1"), (4, 4, "ARG1")}

        # Check that span extraction works with merging.
        spans = metric._extract_spans(indices, merge=True)
        assert spans == {(0, 4, "ARG1")}

        tag_matrix = [['*' for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        tag_matrix[1][1] = "ARG1"
        tag_matrix[3][1] = "ARG1"
        indices = [[self.vocab.get_token_index(
            x, "tags") for x in tag_list] for tag_list in tag_matrix]
        spans = metric._extract_spans(indices, merge=True)
        assert spans == {(0, 3, "ARG1")}

    def test_span_metrics_are_computed_correctly_for_correct_segmentation(self):
        max_span_width = 2
        sent_len = 6
        batch_size = 3
        tags_to_ignore = ['V', '*', 'O']
        metric = NonBioSpanBasedF1Measure(
            self.vocab, "tags", tags_to_ignore)

        gold_spans = [[['*' for _ in range(max_span_width)]
                       for _ in range(sent_len)] for _ in range(batch_size)]

        # B-A1 I-A1 I-A1 V * *
        gold_spans[0][1][1] = 'ARG1'
        gold_spans[0][2][0] = 'ARG1'
        gold_spans[0][3][0] = 'V'

        # B-A1 O V B-A2 B-A2 I-A2
        gold_spans[1][0][0] = 'ARG1'
        gold_spans[1][1][0] = 'O'
        gold_spans[1][2][0] = 'V'
        gold_spans[1][3][0] = 'ARG2'
        gold_spans[1][5][1] = 'ARG2'

        # V O O B-A1 I-A1 *
        gold_spans[2][0][0] = 'V'
        gold_spans[2][2][1] = 'O'
        gold_spans[2][4][1] = 'ARG1'

        arg1 = self.vocab.get_token_index('ARG1', 'tags')
        arg2 = self.vocab.get_token_index('ARG2', 'tags')
        o = self.vocab.get_token_index('O', 'tags')
        # v = self.vocab.get_token_index('V', 'tags')

        mask = torch.ones(batch_size, sent_len)
        mask[0][4] = 0
        mask[0][5] = 0
        mask[2][5] = 0

        gold_indices = [[[self.vocab.get_token_index(
            x, "tags") for x in rows] for rows in ex] for ex in gold_spans]
        gold_tensor = torch.Tensor(gold_indices)

        # Check for case when prediction is the same as gold.
        prediction_tensor = gold_tensor
        metric(prediction_tensor, gold_tensor, mask)

        assert metric._true_positives["ARG1"] == 3  # merged in ex 0
        assert metric._true_positives["ARG2"] == 1  # merged in ex 1

        assert metric._false_negatives["ARG1"] == 0
        assert metric._false_negatives["ARG2"] == 0

        assert metric._false_positives["ARG1"] == 0
        assert metric._false_positives["ARG2"] == 0

        for tag in tags_to_ignore:
            assert tag not in metric._true_positives.keys()
            assert tag not in metric._false_negatives.keys()
            assert tag not in metric._false_positives.keys()

        metric_dict = metric.get_metric()
        np.testing.assert_almost_equal(metric_dict["recall-ARG2"], 1.0)
        np.testing.assert_almost_equal(metric_dict["precision-ARG2"], 1.0)
        np.testing.assert_almost_equal(metric_dict["f1-measure-ARG2"], 1.0)

        np.testing.assert_almost_equal(metric_dict["recall-ARG1"], 1.0)
        np.testing.assert_almost_equal(metric_dict["precision-ARG1"], 1.0)
        np.testing.assert_almost_equal(metric_dict["f1-measure-ARG1"], 1.0)

        np.testing.assert_almost_equal(metric_dict["recall-overall"], 1.0)
        np.testing.assert_almost_equal(metric_dict["precision-overall"], 1.0)
        np.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 1.0)

        # Test that the span measure ignores completely masked sequences by
        # passing a mask with a fully masked row.
        # mask = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                          [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # prediction_tensor[:, 0, 0] = 1
        # prediction_tensor[:, 1, 1] = 1  # (True positive - ARG1
        # prediction_tensor[:, 2, 2] = 1  # *)
        # prediction_tensor[:, 3, 0] = 1
        # prediction_tensor[:, 4, 0] = 1  # (False Negative - ARG2
        # prediction_tensor[:, 5, 0] = 1  # *)
        # prediction_tensor[:, 6, 0] = 1
        # prediction_tensor[:, 7, 1] = 1  # (False Positive - ARG1
        # prediction_tensor[:, 8, 2] = 1  # *)

    def test_span_metrics_are_computed_correctly_for_incorrect_segmentation(self):
        max_span_width = 2
        sent_len = 3
        batch_size = 1
        tags_to_ignore = ['*', 'O']
        metric = NonBioSpanBasedF1Measure(
            self.vocab, "tags", tags_to_ignore)

        star = self.vocab.get_token_index('*', 'tags')
        arg2 = self.vocab.get_token_index('ARG2', 'tags')
        o = self.vocab.get_token_index('O', 'tags')

        gold_spans = [[star for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        gold_spans[0][0] = o
        gold_spans[2][1] = arg2
        gold_tensor = torch.Tensor([gold_spans])

        mask = torch.ones(batch_size, sent_len)

        # B-A2 B-A2 I-A2
        prediction = [[star for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        prediction[0][0] = arg2
        prediction[2][1] = arg2
        prediction_tensor = torch.Tensor([prediction])
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives['ARG2'] == 0
        assert metric._false_positives['ARG2'] == 1
        assert metric._false_negatives['ARG2'] == 1
        metric.reset()

        # B-A2 I-A2 B-O
        prediction = [[star for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        prediction[1][1] = arg2
        prediction[2][0] = o
        prediction_tensor = torch.Tensor([prediction])
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives['ARG2'] == 0
        assert metric._false_positives['ARG2'] == 1
        assert metric._false_negatives['ARG2'] == 1
        metric.reset()

        # B-A2 B-O B-A2
        prediction = [[star for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        prediction[0][0] = arg2
        prediction[1][0] = o
        prediction[2][0] = arg2
        prediction_tensor = torch.Tensor([prediction])
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives['ARG2'] == 0
        assert metric._false_positives['ARG2'] == 2
        assert metric._false_negatives['ARG2'] == 1
        metric.reset()

        # B-A2 B-O I-O
        prediction = [[star for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        prediction[0][0] = arg2
        prediction[2][1] = o
        prediction_tensor = torch.Tensor([prediction])
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives['ARG2'] == 0
        assert metric._false_positives['ARG2'] == 1
        assert metric._false_negatives['ARG2'] == 1
        metric.reset()

        # B-O B-A2 I-A2
        prediction = [[star for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        prediction[0][0] = o
        prediction[2][1] = arg2
        prediction_tensor = torch.Tensor([prediction])
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives['ARG2'] == 1
        assert metric._false_positives['ARG2'] == 0
        assert metric._false_negatives['ARG2'] == 0
        metric.reset()

        # B-O B-A2 B-O
        prediction = [[star for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        prediction[0][0] = o
        prediction[1][0] = arg2
        prediction[2][0] = o
        prediction_tensor = torch.Tensor([prediction])
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives['ARG2'] == 0
        assert metric._false_positives['ARG2'] == 1
        assert metric._false_negatives['ARG2'] == 1
        metric.reset()

        # B-O I-O B-A2
        prediction = [[star for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        prediction[1][1] = o
        prediction[2][0] = arg2
        prediction_tensor = torch.Tensor([prediction])
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives['ARG2'] == 0
        assert metric._false_positives['ARG2'] == 1
        assert metric._false_negatives['ARG2'] == 1
        metric.reset()

        # B-O B-O I-O
        prediction = [[star for _ in range(max_span_width)]
                      for _ in range(sent_len)]
        prediction[0][0] = o
        prediction[2][1] = o
        prediction_tensor = torch.Tensor([prediction])
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives['ARG2'] == 0
        assert metric._false_positives['ARG2'] == 0
        assert metric._false_negatives['ARG2'] == 1
        metric.reset()

    def test_span_f1_can_build_from_params(self):
        params = Params(
            {"type": "non_bio_span_f1", "tag_namespace": "tags", "ignore_classes": ["V"]})
        metric = Metric.from_params(params, self.vocab)
        assert metric._ignore_classes == ["V"]
        assert metric._label_vocabulary == self.vocab.get_index_to_token_vocabulary(
            "tags")
