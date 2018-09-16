from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import ones_like, get_lengths_from_binary_sequence_mask
from allennlp.data.dataset_readers.framenet.ontology_reader import FrameOntology
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric


@Metric.register("non_bio_span_f1")
class NonBioSpanBasedF1Measure(Metric):
    """
    The Conll SRL metrics are based on exact span matching. This metric
    implements span-based precision and recall metrics for a BIO tagging
    scheme. It will produce precision, recall and F1 measures per tag, as
    well as overall statistics. Note that the implementation of this metric
    is not exactly the same as the perl script used to evaluate the CONLL 2005
    data - particularly, it does not consider continuations or reference spans
    as constituents of the original span. However, it is a close proxy, which
    can be helpful for judging model peformance during training.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 tag_namespace: str = "tags",
                 ignore_classes: List[str] = None,
                 ontology_path: str = None) -> None:
        """
        Parameters
        ----------
        vocabulary : ``Vocabulary``, required.
            A vocabulary containing the tag namespace.
        tag_namespace : str, required.
            This metric assumes that a BIO format is used in which the
            labels are of the format: ["B-LABEL", "I-LABEL"].
        ignore_classes : List[str], optional.
            Span labels which will be ignored when computing span metrics.
            A "span label" is the part that comes after the BIO label, so it
            would be "ARG1" for the tag "B-ARG1". For example by passing:

             ``ignore_classes=["V"]``
            the following sequence would not consider the "V" span at index (2, 3)
            when computing the precision, recall and F1 metrics.

            ["O", "O", "B-V", "I-V", "B-ARG1", "I-ARG1"]

            This is helpful for instance, to avoid computing metrics for "V"
            spans in a BIO tagging scheme which are typically not included.
        """
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(
            tag_namespace)
        self._ignore_classes = ignore_classes or []
        self.num_classes = vocabulary.get_vocab_size(tag_namespace)

        if ontology_path is not None:
            self._ontology = FrameOntology(ontology_path)

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 frames: List[str] = None):
        """
        Parameters
        ----------
        num_classes: ``int``, required.
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, max_span_width).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length, max_span_width).
            It must be the same shape as the ``predictions`` tensor.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor of shape (batch_size, sequence_length).
        """
        def score(frame: str, frame_element: str):
            # FN non-core FEs account for 0.5 only
            if frame and self._ontology:
                if frame_element not in self._ontology.core_frame_map[frame]:
                    return 0.5
            return 1.0

        if (gold_labels >= self.num_classes).any():
            raise ConfigurationError("A gold label passed to NonBioSpanBasedF1Measure contains an "
                                     "id >= {}, the number of classes.".format(self.num_classes))

        if mask is None:
            mask = ones_like(gold_labels)

        # Get the data from the Variables.
        predictions, gold_labels, mask = self.unwrap_to_tensors(
            predictions, gold_labels, mask)

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.float()

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]
            frame = None
            if frames:
                frame = frames[i]

            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue
            prediction_spans = self._extract_spans(
                sequence_prediction[:length].tolist(), merge=True)
            gold_spans = self._extract_spans(
                sequence_gold_label[:length].tolist(), merge=True)

            # FN is not to be evaluated for empty gold annotations.
            if not gold_spans and frames:
                continue

            for span in prediction_spans:
                label = span[2]
                if span in gold_spans:
                    self._true_positives[label] += score(frame, label)
                    gold_spans.remove(span)
                else:
                    self._false_positives[label] += score(frame, label)
            # These spans weren't predicted.
            for span in gold_spans:
                label = span[2]
                self._false_negatives[label] += score(frame, label)

    def _extract_spans(self, tag_matrix: List[List[int]], merge: bool = False) -> Set[Tuple[int, int, str]]:
        """
        Given an integer tag sequence corresponding to BIO tags, extracts spans.
        Spans are inclusive and can be of zero length, representing a single word span.
        Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
        as otherwise it is possible to get a perfect precision score whilst still predicting
        ill-formed spans in addition to the correct spans.

        Parameters
        ----------
        tag_tensor : List[List[int]], required.
            The integer class labels for a sequence.

        Returns
        -------
        spans : Set[Tuple[int, int, str]]
            The typed, extracted spans from the sequence, in the format ((span_start, span_end), label).
            Note that the label `does not` contain any BIO tag prefixes.
        """
        spans = set()
        span_start = 0
        span_end = 0

        for span_end, diff_list in enumerate(tag_matrix):
            for diff, tag_id in enumerate(diff_list):
                # Actual tag.
                tag_string = self._label_vocabulary[tag_id]
                # We don't care about tags we are told to ignore, so we do nothing.
                if tag_string in self._ignore_classes:
                    continue

                if span_end - diff < 0:
                    continue
                span_start = span_end - diff
                spans.add((span_start, span_end, tag_string))
        if merge:
            return self.merge_neighboring_spans(spans)
        return spans

    def merge_neighboring_spans(self, labeled_spans: Set[Tuple[int, int, str]]):
        """
        Merges adjacent spans with the same label, ONLY for the prediction (to encounter spurious ambiguity).

        Returns
        -------
        List[Tuple[int, int, str]]
            where each tuple represents start, end and label of a span.
        """
        # Full empty prediction.
        if not labeled_spans:
            return labeled_spans
        # Create a sorted copy.
        args = sorted([x for x in list(labeled_spans)])
        start, end, label = args[0]
        for arg in args[1:]:
            if arg[2] == label and arg[0] == end+1:
                labeled_spans.remove(arg)
                labeled_spans.remove((start, end, label))
                labeled_spans.add((start, arg[1], label))
                end = arg[1]
            else:
                start, end, label = arg
        return labeled_spans

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values(
                                                              )),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / \
            float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / \
            float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
