"""
Semi-Markov Conditional random field
"""
import logging
from typing import Dict, List, Tuple

import sys
import torch
from torch.autograd import Variable

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import logsumexp, ones_like

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SemiMarkovConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute the log-likelihood
    of its inputs assuming a 0th-order semi-Markov conditional random field model.
    The semi comes from the fact that there is no Markovian order inside of a segment.

    See, e.g. http://www.cs.cmu.edu/~wcohen/postscript/semiCRF.pdf

    Parameters
    ----------
    num_tags : int, required
        The number of tags.
    default_tag : int, required
        Index of tag '*' denoting the spans which are invalid.
    max_span_width : int, required.
        Maximum allowed width of a span.
    """

    def __init__(self,
                 num_tags: int,
                 default_tag: int,
                 max_span_width: int,
                 outside_span_tag: int = None,
                 loss_type: str = "logloss",
                 false_positive_penalty: float = 1.0,
                 false_negative_penalty: float = 1.0) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.max_span_width = max_span_width
        self.default_tag = default_tag
        self.outside_span_tag = outside_span_tag
        self.loss_type = loss_type
        self.false_positive_penalty = false_positive_penalty
        self.false_negative_penalty = false_negative_penalty

    def _input_likelihood(self,
                          logits: torch.Tensor,
                          text_mask: torch.Tensor,
                          cost: torch.Tensor,
                          tag_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible segmentations.
        Side effects: suffers from spurious ambiguity.

        Parameters
        ----------
        logits : shape (batch_size, sequence_length, max_span_width, num_tags)
        text_mask : shape (batch_size, sequence_length)
        span_mask : shape (batch_size, sequence_length, max_span_width)
        tag_mask : shape (batch_size, num_tags)
        cost : shape (batch_size, sequence_length, max_span_width, num_tags)
        """
        batch_size, sequence_length, max_span_width, num_tags = logits.size()
        # This way of masking introduces nan gradients, avoid!
        # span_mask = span_mask.view(batch_size, -1, self.maximum_segment_length, 1) * text_mask.view(
        #     batch_size, -1, 1, 1)
        # logits = logits + torch.log(span_mask.float())
        if tag_mask is None:
            tag_mask = Variable(torch.ones(batch_size, num_tags).cuda())
        else:
            tmask_sum = torch.sum(tag_mask, 1).data
            assert (tmask_sum > 0).all()

        # shape: (sequence_length, max_span_width, batch_size, num_tags)
        logits = logits.transpose(0, 1).contiguous()
        logits = logits.transpose(1, 2).contiguous()

        cost = cost.transpose(0, 1).contiguous()
        cost = cost.transpose(1, 2).contiguous()

        # Create a mask to ignore the dummy tag '*' denoting not a span.
        default_tag_mask = torch.zeros(num_tags).cuda()
        default_tag_mask[self.default_tag] = float("-inf")
        default_tag_mask = Variable(default_tag_mask.view(1, 1, -1))

        # Move to log space.
        tag_mask = torch.log(tag_mask).view(
            1, batch_size, num_tags) + default_tag_mask

        # Initial alpha is the (sequence_length, batch_size) tensor of likelihoods containing the
        # logits for the first dummy timestep.
        alpha = Variable(torch.cuda.FloatTensor([
            [0.0 for _ in range(batch_size)]]), requires_grad=True)

        # For each j we compute logits for all the segmentations of length j.
        for j in range(sequence_length):
            # Depending on where the span ends, i.e. j, the maximum width of the spans considered changes.
            width = max_span_width
            if j < max_span_width - 1:
                width = j + 1

            # Reverse the alpha so it gets added to the correct logits.
            idx = Variable(torch.cuda.LongTensor(
                [i for i in range(j, j - width, -1)]))
            reversed_alpha = alpha.index_select(dim=0, index=idx)
            # Tensorize and broadcast along the max_span_width dimension.
            broadcast_alpha = reversed_alpha.view(width, batch_size, 1)

            # shape: (max_span_width, batch_size, num_tags)
            logits_at_j = logits[j]
            start_indices = Variable(torch.cuda.LongTensor(range(width)))
            span_factors = logits_at_j.index_select(dim=0, index=start_indices)
            span_costs = cost[j].index_select(dim=0, index=start_indices)

            # Logsumexp the scores over the num_tags axis.
            alpha_along_arglabels = logsumexp(
                broadcast_alpha + span_factors + span_costs + tag_mask)

            # Logsumexp the scores over the max_span_width axis.
            alpha_at_j = logsumexp(alpha_along_arglabels,
                                   dim=0).view(1, batch_size)
            alpha = torch.cat([alpha, alpha_at_j], dim=0)

        # Get the last positions for all alphas in the batch.
        actual_lengths = torch.sum(text_mask, dim=1).view(1, batch_size)
        # Finally we return the alphas along the last "valid" positions.
        # shape: (batch_size)
        partition = alpha.gather(dim=0, index=actual_lengths)
        return partition

    def _joint_likelihood(self,
                          logits: torch.Tensor,
                          tags: torch.Tensor,
                          mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)

        Parameters
        ----------
        logits : shape (batch_size, sequence_length, max_span_width, num_tags)
        tags : shape (batch_size, sequence_length, max_span_width)
        mask : shape (batch_size, sequence_length)
        """
        batch_size, sequence_length, _, _ = logits.shape

        # Transpose to shape: (sequence_length, max_span_width, batch_size, num_tags)
        logits = logits.transpose(0, 1).contiguous()
        logits = logits.transpose(1, 2).contiguous()
        # Transpose to shape: (sequence_length, batch_size)
        mask = mask.float().transpose(0, 1).contiguous()
        # Transpose to shape: (sequence_length, max_span_width, batch_size)
        tags = tags.transpose(0, 1).contiguous()
        tags = tags.transpose(1, 2).contiguous()

        default_tags = Variable(
            self.default_tag * torch.ones(batch_size).long().cuda())

        numerator = 0.0
        # Add up the scores for the observed segmentations
        for j in range(sequence_length):
            # # shape: (max_seg_len, batch_size)
            # batched_tags = tags[j]  # .transpose(0, 1).contiguous()
            # # shape: (max_seg_len, batch_size, num_tags)
            # batched_logits = logits[j]  # .transpose(0, 1).contiguous()
            for d in range(min(self.max_span_width, sequence_length)):
                current_tag = tags[j][d]
                # Ignore tags for invalid spans.
                valid_tag_mask = (current_tag != default_tags).float()
                # Reshape for gather operation to follow.
                current_tag = current_tag.view(batch_size, 1)
                # The score for using current_tag
                emit_score = logits[j][d].gather(dim=1, index=current_tag).squeeze(
                    1) * valid_tag_mask * mask[j]
                numerator += emit_score
        return numerator

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor,
                tag_mask: Variable = None,
                average_batches: bool = True) -> torch.Tensor:
        """
        Computes the log likelihood.
        Parameters
        ----------
        inputs : shape (batch_size, sequence_length, max_span_width, num_tags)
        tags : shape (batch_size, sequence_length, max_span_width)
        mask : shape (batch_size, sequence_length)
        tag_mask : shape (batch_size, num_tags)
        """
        # pylint: disable=arguments-differ
        batch_size = inputs.size(0)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        if self.loss_type == "roc":
            cost = self._get_recall_oriented_cost(tags)
        elif self.loss_type == "hamming":
            cost = self._get_hamming_cost(tags)
        elif self.loss_type == "logloss":
            zeroes = 1 - ones_like(inputs)
            cost = zeroes
        else:
            raise ConfigurationError(
                "invalid loss type {} - use roc, hamming or logloss".format(self.loss_type))

        log_denominator = self._input_likelihood(logits=inputs,
                                                 text_mask=mask,
                                                 tag_mask=tag_mask,
                                                 cost=cost)
        log_loss = log_numerator - log_denominator
        if self.loss_type == "roc":
            log_loss = log_loss - self.false_negative_penalty * \
                self._get_labeled_spans_count(tags)

        batch_loss = torch.sum(log_loss)
        if average_batches:
            batch_loss = batch_loss / batch_size

        if batch_loss.data[0] > 0.0:
            max_log_loss, _ = torch.max(log_loss, -1)
            logger.info("WARNING: invalid log loss = %f", max_log_loss.data[0])
        # assert batch_loss.data[0] <= 0.0
        return batch_loss, log_numerator

    def viterbi_tags(self,
                     logits: Variable,
                     mask: Variable,
                     tag_masks: Variable = None) -> List[List[int]]:
        """
        Iterates through the batch and uses viterbi algorithm to find most likely tags
        for the given inputs.

        Returns
        -------
        all_tags : torch.Tensor
            shape (batch_size, sequence_length, max_span_width)
        all_scores : torch.Tensor
            shape (batch_size, sequence_length, max_span_width, num_tags)
        """
        batch_size, max_seq_length, max_span_width, num_classes = logits.size()

        if tag_masks is None:
            tag_masks = Variable(torch.ones(batch_size, num_classes).cuda())

        # Get the tensors out of the variables
        logits, mask, tag_masks = logits.data, mask.data, tag_masks.data
        sequence_lengths = torch.sum(mask, dim=-1)

        all_tags = []
        all_scores = []
        for logits_ex, tag_mask, sequence_length in zip(logits, tag_masks, sequence_lengths):
            # We need to maintain this length, because all_tags needs to be of the same size as tags
            tags = [[self.default_tag for _ in range(max_span_width)]
                    for _ in range(max_seq_length)]
            scores = [[[float("-inf") for _ in range(num_classes)] for _ in range(max_span_width)]
                      for _ in range(max_seq_length)]

            # We pass the logits to ``viterbi_decode``.
            viterbi_path, viterbi_score = self.viterbi_decode(
                logits_ex[:sequence_length], tag_mask)

            tags[:len(viterbi_path)] = viterbi_path
            scores[:len(viterbi_score)] = viterbi_score

            # shape: (batch_size, max_seq_length, max_span_width)
            all_tags.append(tags)
            # shape: (batch_size, max_seq_length, max_span_width, num_classes)
            all_scores.append(scores)

        return torch.Tensor(all_tags), torch.Tensor(all_scores)

    def viterbi_decode(self, logits: torch.Tensor, tag_mask: torch.Tensor):
        """
        Perform 0-th order Semi-Markov Viterbi decoding in log space over a sequence given
        a matrix of shape (sequence_length, span_width, num_tags) specifying unary potentials
        for possible tags per span in the sequence.

        Parameters
        ----------
        logits : torch.Tensor, required.
            A tensor of shape (sequence_length, span_width, num_tags) representing scores for
            a set of tags over a given sequence.
        tag_mask: torch.Tensor, required.
            shape (num_tags)

        Returns
        -------
        viterbi_path : List[List[int]]
            The tag indices of the maximum likelihood tag sequence.
        viterbi_score : torch.Tensor
            shape (sequence_length, max_span_width, num_tags)
            The score of the viterbi path.
        """
        sequence_length, max_span_width, num_classes = list(logits.size())

        tag_mask = torch.log(tag_mask).view(1, num_classes)

        alpha = [float('-inf')
                 for _ in range(sequence_length)]  # shape : [sequence_length]
        backpointers = [(None, None)
                        for _ in range(sequence_length)]  # shape : [sequence_length]

        # Evaluate the scores for all possible paths.
        for j in range(sequence_length):
            width = max_span_width
            if j < max_span_width - 1:
                width = j + 1

            # Find the best labels (and their scores) for all spans ending at j
            start_indices = torch.cuda.LongTensor(range(width))
            span_factors = logits[j].index_select(0, start_indices)
            best_span_factors, best_labels = torch.max(
                span_factors + tag_mask, -1)

            # Add a dummy dimension to alpha (for position -1) and reverse it.
            extended_alpha = [0.0] + alpha
            broadcast_alpha = torch.cuda.FloatTensor(
                extended_alpha[j + 1 - width:j + 1][::-1])

            # Add pairwise potentials to current scores.
            summed_potentials = broadcast_alpha + best_span_factors
            best_score, best_difference = torch.max(summed_potentials, -1)

            # Reverse this, since it corresponds to reversed idx.
            best_difference = int(best_difference)
            alpha[j] = float(best_score)
            backpointers[j] = (best_labels[best_difference], best_difference)

        # Construct the most likely sequence backwards.
        viterbi_path = [[self.default_tag for _ in range(max_span_width)]
                        for _ in range(sequence_length)]
        # Also, keep track of the span indices and the associated tag.
        viterbi_spans = {}
        # Also construct the best scoring tensor (for evaluation, not quite necessary).
        viterbi_score = torch.Tensor([[[float("-inf") for _ in range(sequence_length)]
                                       for _ in range(max_span_width)] for _ in range(num_classes)])
        viterbi_score[self.default_tag] = 0.0
        viterbi_score = viterbi_score.transpose(0, 2).tolist()

        # Start from the end.
        span_end = sequence_length - 1
        while span_end >= 0:
            label, width = backpointers[span_end]
            viterbi_path[span_end][width] = label
            viterbi_spans[(span_end - width, span_end)] = label
            if label != self.default_tag:
                viterbi_score[span_end][width][self.default_tag] = float(
                    "-inf")
            viterbi_score[span_end][width][label] = alpha[span_end]
            span_end = span_end - width - 1

        return viterbi_path, viterbi_score

    def convert_spans_into_sequence_of_tags(self, viterbi_spans: Dict[Tuple[int, int], int],
                                            sequence_length: int,
                                            num_classes: int) -> List[int]:
        tag_sequence = [None for _ in range(sequence_length)]
        tag_indicators = [
            [0.0 for _ in range(num_classes)] for _ in range(sequence_length)]
        for span in viterbi_spans:
            for position in range(span[0], span[1] + 1):
                # Make sure that the current position is not already assigned.
                assert not tag_sequence[position]
                tag_sequence[position] = viterbi_spans[span]
                tag_indicators[position][viterbi_spans[span]] = 1.0
        # Make sure every position got a tag.
        assert None not in tag_sequence
        return tag_indicators

    def merge_spans(self, tag_sequence: List[int]) -> List[List[int]]:
        spans = [[self.default_tag for _ in range(self.max_span_width)]
                 for _ in range(len(tag_sequence))]

        start_span = 0
        current_tag = tag_sequence[0]
        for pos, tag in enumerate(tag_sequence[1:], 1):
            width = pos - start_span
            if tag != current_tag:
                width = pos - 1 - start_span
                spans[pos - 1][width] = current_tag
                start_span = pos
                current_tag = tag
                width = pos - start_span
            # Maximum allowed width.
            elif width == self.max_span_width - 1:
                spans[pos][width] = current_tag
                start_span = pos + 1
                if pos + 1 < len(tag_sequence):
                    current_tag = tag_sequence[pos + 1]
        spans[len(tag_sequence) - 1][len(tag_sequence) -
                                     1 - start_span] = tag_sequence[-1]
        return spans

    def _get_hamming_cost(self, tags: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, max_span_width = tags.size()
        # Make tags the same dim as required cost, for indexing.
        tags = tags.unsqueeze(dim=-1)
        zeros = Variable(torch.zeros(batch_size, sequence_length,
                                     max_span_width, self.num_tags).float().cuda())
        scattered_tags = zeros.scatter_(-1, tags, 1)
        cost = 1 - scattered_tags

        # Now mask out the cost assigned to places without a real tag ~ "*"
        default_tags_mask = 1-tags.eq(self.default_tag).float()
        cost = cost * default_tags_mask

        return self.false_positive_penalty * cost

    def _get_simple_recall_cost(self, tags: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, max_span_width = tags.size()
        # Make tags the same dim as required cost, for indexing.
        tags = tags.unsqueeze(dim=-1)
        zeros = Variable(torch.zeros(batch_size, sequence_length,
                                     max_span_width, self.num_tags).float().cuda())
        scattered_tags = zeros.scatter_(-1, tags, 1)
        cost = 1 - scattered_tags

        # Now mask out the cost assigned to tags that are either "*" or outside span.
        irrelevant_tags = tags.eq(
            self.default_tag) | tags.eq(self.outside_span_tag)
        irrelevant_tags_mask = 1-irrelevant_tags.float()
        cost = cost * irrelevant_tags_mask

        return self.false_negative_penalty * cost

    def _get_recall_oriented_cost(self, tags: torch.Tensor):
        batch_size, sequence_length, max_span_width = tags.size()
        # Make tags the same dim as required cost, for indexing.
        tags = tags.unsqueeze(dim=-1)
        zeros = Variable(torch.zeros(batch_size, sequence_length,
                                     max_span_width, self.num_tags).float().cuda())
        scattered_tags = zeros.scatter_(-1, tags, 1)
        cost = 1 - scattered_tags

        # False Positives
        # Now mask out the cost assigned to places without a real tag ~ "*"
        default_tags_mask = 1-tags.eq(self.default_tag).float()
        fp = cost * default_tags_mask
        # Masking out all the "O"s
        fp = fp.index_fill_(-1,
                            Variable(torch.cuda.LongTensor([self.outside_span_tag])), 0)
        fp = self.false_positive_penalty * fp

        # False Negatives
        irrelevant_tags = tags.eq(
            self.default_tag) | tags.eq(self.outside_span_tag)
        irrelevant_tags_mask = 1-irrelevant_tags.float()
        fn = - self.false_negative_penalty * cost * irrelevant_tags_mask
        return fp + fn

    def _get_labeled_spans_count(self, tags: torch.Tensor):
        batch_size, sequence_length, max_span_width = tags.size()
        # Make tags the same dim as required cost, for indexing.
        tags = tags.unsqueeze(dim=-1)
        zeros = Variable(torch.zeros(batch_size, sequence_length,
                                     max_span_width, self.num_tags).float().cuda())
        scattered_tags = zeros.scatter_(-1, tags, 1)

        irrelevant_tags = tags.eq(
            self.default_tag) | tags.eq(self.outside_span_tag)
        irrelevant_tags_mask = 1-irrelevant_tags.float()

        total_relevant_labels = torch.sum(
            torch.sum(torch.sum(scattered_tags * irrelevant_tags_mask, -1), -1), -1)
        return total_relevant_labels
