# pylint: disable=no-self-use,invalid-name
import itertools
import math

from pytest import approx
import torch
from torch.autograd import Variable

from allennlp.modules import SemiMarkovConditionalRandomField
from allennlp.common.testing import AllenNlpTestCase


class TestSemiMarkovConditionalRandomField(AllenNlpTestCase):
    """
    Tests for SemiMarkovConditionalRandomField
    """

    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.sentence_len = 3
        self.max_segment_length = 2
        self.num_tags = 3  # for [A, B, *]
        ninf = -500.0
        self.logits = Variable(torch.Tensor([
            [[[0.20, 0.35, 0.00], [ninf, ninf, ninf]],
             [[0.30, 0.16, 0.10], [1.20, 0.05, 0.00]],
             [[0.10, 1.80, ninf], [0.00, 0.10, 0.00]]],

            [[[0.40, 2.00, 0.00], [ninf, ninf, ninf]],
             [[1.80, 0.40, 0.00], [0.30, 0.10, 0.00]],
             [[ninf, ninf, ninf], [ninf, ninf, ninf]]]
        ]).cuda())
        self.tags_sequence = [
            [[2, 2], [2, 0], [1, 2]],
            [[1, 2], [0, 2], [2, 2]]
        ]
        self.span_mask = Variable(torch.LongTensor([
            [[1, 0], [1, 1], [1, 1]],
            [[1, 0], [1, 1], [0, 0]]
        ]).cuda())
        self.default_tag = 2
        self.outside_span_tag = 1
        self.tags_tensor = Variable(
            torch.LongTensor(self.tags_sequence).cuda())
        self.mask = Variable(torch.LongTensor([[1, 1, 1], [1, 1, 0]]).cuda())

        # Use the Semi CRF module to compute the log_likelihood
        self.semi_crf = SemiMarkovConditionalRandomField(num_tags=self.num_tags,
                                                         max_span_width=self.max_segment_length,
                                                         default_tag=self.default_tag,
                                                         outside_span_tag=self.outside_span_tag,
                                                         false_negative_penalty=5.0,
                                                         false_positive_penalty=1.0)

    def score(self, logits, tags):
        """
        Computes the likelihood score for the given sequence of tags,
        given the provided logits (and the transition weights in the CRF model)
        """
        total = 0.
        # Add in the logits for the observed tags
        # shape : [sentence_len * max_span_width, num_tags]
        logit_reshaped = logits.view(-1, self.num_tags)
        tag_reshaped = tags.view(-1)
        mask = tag_reshaped != self.default_tag
        for i, t in enumerate(tag_reshaped):
            if mask[i].data[0]:
                total += logit_reshaped[i][t]
        return total

    def create_tags_tensor_from_sequence(self, tag_sequence):
        assert len(tag_sequence) <= self.sentence_len
        tags = [[self.default_tag for _ in range(self.max_segment_length)]
                for _ in range(self.sentence_len)]

        start_span = 0
        for current in range(1, len(tag_sequence)):
            current_position_tagged = False
            if tag_sequence[current] != tag_sequence[current - 1]:
                tags[current - 1][current - 1 -
                                  start_span] = tag_sequence[current - 1]
                start_span = current
            elif current - start_span == self.max_segment_length - 1:
                tags[current][current -
                              start_span] = tag_sequence[current]
                current_position_tagged = True
                start_span = current + 1
            if current == len(tag_sequence) - 1 and not current_position_tagged:
                tags[current][current -
                              start_span] = tag_sequence[current]

        return Variable(torch.LongTensor(tags).cuda())

    def get_manual_log_likelihood(self):
        manual_log_likelihood = 0.0
        sentence_lengths = list(torch.sum(self.mask, -1).data)

        spurious_scores = [[([0, 0, 0], 0.6),
                            ([0, 0, 0], 0.2),
                            ([0, 0, 1], 2.3),
                            ([0, 1, 1], 2.16),
                            ([1, 0, 0], 0.75),
                            ([1, 1, 0], 0.61),
                            ([1, 1, 1], 0.45),
                            ([1, 1, 1], 2.31)],
                           [([0, 0], 2.2),
                            ([1, 1], 2.4)]
                           ]
        numerators = []
        # Computing the log likelihood by enumerating all possible sequences.
        for logit, tag, sentence_length, spurious_score in zip(self.logits, self.tags_tensor, sentence_lengths, spurious_scores):
            numerator = self.score(logit, tag)
            numerators.append(numerator.data[0])

            all_tags = [(seq, self.create_tags_tensor_from_sequence(seq))
                        for seq in itertools.product(range(self.num_tags - 1), repeat=sentence_length)]
            all_scores = [(tag_tensor[0], self.score(logit, tag_tensor[1]))
                          for tag_tensor in all_tags]

            denominator = math.log(
                sum([math.exp(score[1]) for score in spurious_score + all_scores]))
            manual_log_likelihood += numerator - denominator

        return numerators, manual_log_likelihood.data[0]

    def test_forward(self):
        """
        Tests forward without any tag mask.
        """
        assert torch.Size([self.batch_size, self.sentence_len,
                           self.max_segment_length, self.num_tags]) == self.logits.size()

        ll, numerator = self.semi_crf(self.logits, self.tags_tensor, self.mask)
        expected_numerator, expected_ll = self.get_manual_log_likelihood()

        print("\nActual log likelihood =", ll.data[0],
              "\nExpected log likelihood =", expected_ll)
        print("Actual numerator =", numerator.data.tolist(),
              "\nExpected numerator =", expected_numerator)
        assert expected_ll == approx(ll.data[0])
        assert expected_numerator == numerator.data.tolist()

    def test_viterbi_tags(self):
        """
        Test viterbi with back-pointers.
        # TODO(Swabha): Test viterbi_scores.
        """
        viterbi_tags, _ = self.semi_crf.viterbi_tags(self.logits, self.mask)
        assert viterbi_tags.tolist() == self.tags_sequence

    def test_viterbi_tags_merging(self):
        # TODO(Swabha): Test a case where spans are being merged.
        pass

    def test_forward_with_tag_mask(self):
        # TODO(Swabha): Might need a larger number of tags.
        pass

    def test_hamming_cost(self):
        hamming_cost = [
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1]], [[1, 0, 1], [0, 0, 0]]],
            [[[1, 0, 1], [0, 0, 0]], [[0, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
        ]
        assert self.semi_crf._get_hamming_cost(
            self.tags_tensor).data.tolist() == hamming_cost

    def test_simple_recall_cost(self):
        recall_cost = [
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 5]], [[0, 0, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 0, 0]], [[0, 5, 5], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
        ]
        assert self.semi_crf._get_simple_recall_cost(
            self.tags_tensor).data.tolist() == recall_cost

    def test_recall_oriented_cost(self):
        cost = self.semi_crf._get_recall_oriented_cost(self.tags_tensor)
        gold_cost = self.semi_crf._joint_likelihood(
            cost, self.tags_tensor, self.mask)
        print("\ncost:\n", cost.data.tolist())
        print("cost for gold tags:\n", gold_cost.data.tolist())
        # TODO (Swabha): test cost for a different segmentation...

        num_labeled = self.semi_crf._get_labeled_spans_count(self.tags_tensor)
        print("# non-O labeled tags in batch:\n", num_labeled.data.tolist())
        # TODO (Swabha): test num-labeled for a different batch...

    def test_roc_loss(self):
        # TODO(Swabha)
        pass
