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

        # These will hold unlabeled span counts.
        self._unlabeled_true_positives: int = 0
        self._unlabeled_false_positives: int = 0
        self._unlabeled_false_negatives: int = 0

        # These will hold partial match counts.
        self._partial_true_positives: int = 0
        self._partial_false_positives: int = 0
        self._partial_false_negatives: int = 0

        # These will hold width-wise span counts.
        self._width_tp: Dict[int, int] = defaultdict(int)
        self._width_fp: Dict[int, int] = defaultdict(int)
        self._width_fn: Dict[int, int] = defaultdict(int)

        # These will hold width-wise span counts.
        self._dist_tp: Dict[int, int] = defaultdict(int)
        self._dist_fp: Dict[int, int] = defaultdict(int)
        self._dist_fn: Dict[int, int] = defaultdict(int)

        self._gold_spans: List[Set[Tuple[int, int, str]]] = []
        self._predicted_spans: List[Set[Tuple[int, int, str]]] = []

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 frames: List[str] = None,
                 target_indices: torch.Tensor = None):
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
        if (gold_labels >= self.num_classes).any():
            raise ConfigurationError("A gold label passed to NonBioSpanBasedF1Measure contains an "
                                     "id >= {}, the number of classes.".format(self.num_classes))

        if mask is None:
            mask = ones_like(gold_labels)

        # Get the data from the Variables.
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.float()

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            gold_sequence = gold_labels[i, :]
            predicted_sequence = argmax_predictions[i, :]

            length = sequence_lengths[i]
            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue

            gold_spans = self._extract_spans(gold_sequence[:length].tolist(), merge=True)
            predicted_spans = self._extract_spans(predicted_sequence[:length].tolist(), merge=True)

            self._gold_spans.append(gold_spans)
            self._predicted_spans.append(predicted_spans)

            frame = None
            if frames is not None:
                frame = frames[i]

            # FN is not to be evaluated for empty gold annotations.
            if not gold_spans and frame:
                continue

            self._get_labeled_evaluation(gold_spans, predicted_spans, frame)
            self._get_unlabeled_evaluation(gold_spans, predicted_spans, frame)
            self._get_partial_match_evaluation(gold_spans, predicted_spans, frame)
            self._get_width_wise_labeled_evaluation(gold_spans, predicted_spans, frame)

            if target_indices is not None:
                target_index = target_indices[i][0].data[0]
                self._get_distance_wise_labeled_evaluation(gold_spans, predicted_spans, target_index, frame)

    def _score(self, frame: str, frame_element: str):
        # FN non-core FEs account for 0.5 only
        if frame and self._ontology:
            if frame_element not in self._ontology.core_frame_map[frame]:
                return 0.5
        return 1.0

    def _get_labeled_evaluation(self, gold_spans, predicted_spans, frame=None):
        gold_spans_copy = [span for span in gold_spans]
        for span in predicted_spans:
            label = span[2]
            if span in gold_spans:
                self._true_positives[label] += self._score(frame, label)
                gold_spans_copy.remove(span)
            else:
                self._false_positives[label] += self._score(frame, label)
        # These spans weren't predicted.
        for span in gold_spans_copy:
            label = span[2]
            self._false_negatives[label] += self._score(frame, label)

    def _get_unlabeled_evaluation(self, gold_spans, predicted_spans, frame=None):
        unlabeled_gold_spans = [(span[0], span[1]) for span in gold_spans]

        for span in predicted_spans:
            label = span[2]
            unlabeled_span = (span[0], span[1])
            if unlabeled_span in unlabeled_gold_spans:
                self._unlabeled_true_positives += self._score(frame, label)
                unlabeled_gold_spans.remove(unlabeled_span)
            else:
                self._unlabeled_false_positives += self._score(frame, label)

        # These spans weren't predicted.
        for span in gold_spans:
            label = span[2]
            unlabeled_span = (span[0], span[1])
            if unlabeled_span in unlabeled_gold_spans:
                self._unlabeled_false_negatives += self._score(frame, label)

    def _get_tokenwise_evaluation(self, gold_spans, predicted_spans, length, frame=None):
        def get_seq(spans):
            seq = ["-"] * length
            for span in spans:
                start, end, label = span
                for i in range(start, end+1):
                    seq[i] = label
            return seq

        gold_seq = get_seq(gold_spans)
        pred_seq = get_seq(predicted_spans)

        for pred_label, gold_label in zip(pred_seq, gold_seq):
            if pred_label == gold_label == "-":
                continue
            elif pred_label == gold_label != "-":
                self._partial_true_positives += self._score(frame, pred_label)
            elif pred_label != gold_label and pred_label != "-" and gold_label != "-":
                self._partial_false_negatives += self._score(frame, gold_label)
                self._partial_false_positives += self._score(frame, pred_label)
            elif pred_label == "-" and gold_label != "-":
                self._partial_false_negatives += self._score(frame, gold_label)
            elif gold_label == "-" and pred_label != "-":
                self._partial_false_positives += self._score(frame, pred_label)
            else:
                print("unknown case for", pred_label, gold_label)
                raise AttributeError

    def _get_partial_match_evaluation(self, gold_spans, predicted_spans, frame=None):
        gold_spans_dict = {span[2]: (span[0], span[1]) for span in gold_spans}
        for span in predicted_spans:
            start, end, label = span
            if label not in gold_spans_dict:
                self._partial_false_positives += self._score(frame, label)
                continue
            gstart, gend = gold_spans_dict[label]
            prange = set(range(start, end+1))
            grange = set(range(gstart, gend+1))
            if prange.intersection(grange):
                self._partial_true_positives += self._score(frame, label)
                gold_spans_dict.pop(label)
            else:
                self._partial_false_positives += self._score(frame, label)
        for glabel in gold_spans_dict:
            self._partial_false_negatives += self._score(frame, glabel)

    def _get_width_wise_labeled_evaluation(self, gold_spans, predicted_spans, frame=None):
        def _get_bin(width):
            # if width in [0, 1, 2, 3, 4, 5]:
            #     return width
            # elif 5 < width <= 10:
            #     return 10
            # elif 10 < width <= 15:
            #     return 15
            # elif 15 < width <= 25:
            #     return 25
            # elif 25 < width <= 40:
            #     return 40
            # else:
            #     return 100
            return width

        gold_spans_copy = [span for span in gold_spans]
        for span in predicted_spans:
            label = span[2]
            width_bin = _get_bin(span[1]-span[0])
            if span in gold_spans:
                self._width_tp[width_bin] += self._score(frame, label)
                gold_spans_copy.remove(span)
            else:
                self._width_fp[width_bin] += self._score(frame, label)
        # These spans weren't predicted.
        for span in gold_spans_copy:
            label = span[2]
            width_bin = _get_bin(span[1]-span[0])
            self._width_fn[width_bin] += self._score(frame, label)

    def _get_distance_wise_labeled_evaluation(self, gold_spans, predicted_spans, target_index, frame=None):
        gold_spans_copy = [span for span in gold_spans]
        for span in predicted_spans:
            label = span[2]
            if span in gold_spans_copy:
                self._dist_tp[abs(span[1]-target_index)] += self._score(frame, label)
                gold_spans_copy.remove(span)
            else:
                self._dist_fp[abs(span[1]-target_index)] += self._score(frame, label)
        # These spans weren't predicted.
        for span in gold_spans_copy:
            label = span[2]
            self._dist_fn[abs(span[1]-target_index)] += self._score(frame, label)

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
        sorted_spans = sorted([x for x in list(labeled_spans)])
        prev_start, prev_end, prev_label = sorted_spans[0]
        for span in sorted_spans[1:]:
            if span[2] == prev_label and span[0] == prev_end+1:
                # Merge these two spans.
                labeled_spans.remove(span)
                labeled_spans.remove((prev_start, prev_end, prev_label))
                labeled_spans.add((prev_start, span[1], prev_label))
                prev_end = span[1]
            else:
                prev_start, prev_end, prev_label = span
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
        all_metrics = {}

        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
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

        # Compute unlabeled metrics.
        u_precision, u_recall, u_f1_measure = self._compute_metrics(self._unlabeled_true_positives,
                                                                    self._unlabeled_false_positives,
                                                                    self._unlabeled_false_negatives)
        all_metrics["U-p"] = u_precision
        all_metrics["U-r"] = u_recall
        all_metrics["U-f1"] = u_f1_measure

        # Compute partial metrics.
        t_precision, t_recall, t_f1_measure = self._compute_metrics(self._partial_true_positives,
                                                                    self._partial_false_positives,
                                                                    self._partial_false_negatives)
        all_metrics["part-p"] = t_precision
        all_metrics["part-r"] = t_recall
        all_metrics["part-f1"] = t_f1_measure

        # Compute width-wise metrics.
        all_wtags: Set[int] = set()
        all_wtags.update(self._width_tp.keys())
        all_wtags.update(self._width_fp.keys())
        all_wtags.update(self._width_fn.keys())
        for tag in all_wtags:
            precision, recall, f1_measure = self._compute_metrics(self._width_tp[tag],
                                                                  self._width_fp[tag],
                                                                  self._width_fn[tag])
            precision_key = "W-p-{}".format(tag)
            recall_key = "W-r-{}".format(tag)
            f1_key = "W-f1-{}".format(tag)
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute distance-wise metrics.
        all_dtags: Set[int] = set()
        all_dtags.update(self._dist_tp.keys())
        all_dtags.update(self._dist_fp.keys())
        all_dtags.update(self._dist_fn.keys())
        for tag in all_dtags:
            precision, recall, f1_measure = self._compute_metrics(self._dist_tp[tag],
                                                                  self._dist_fp[tag],
                                                                  self._dist_fn[tag])
            precision_key = "D-p-{}".format(tag)
            recall_key = "D-r-{}".format(tag)
            f1_key = "D-f1-{}".format(tag)
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        all_metrics["gold_spans"] = self._gold_spans
        all_metrics["predicted_spans"] = self._predicted_spans

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

        self._unlabeled_true_positives = 0
        self._unlabeled_false_positives = 0
        self._unlabeled_false_negatives = 0

        self._partial_false_negatives = 0
        self._partial_false_positives = 0
        self._partial_true_positives = 0

        self._width_tp = defaultdict(int)
        self._width_fp = defaultdict(int)
        self._width_fn = defaultdict(int)

        self._dist_tp = defaultdict(int)
        self._dist_fp = defaultdict(int)
        self._dist_fn = defaultdict(int)

        self._gold_spans = []
        self._predicted_spans = []