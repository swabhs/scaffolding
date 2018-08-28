from typing import Dict, List, TextIO, Optional

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.token_embedders import Embedding
from allennlp.modules import TimeDistributed
from allennlp.nn import util


def compute_span_representations(max_span_width: int,
                                 encoded_text: torch.FloatTensor,
                                 target_index: torch.IntTensor,
                                 span_starts: torch.IntTensor,
                                 span_ends: torch.IntTensor,
                                 span_width_embedding: Embedding,
                                 span_direction_embedding: Embedding,
                                 span_distance_embedding: Embedding,
                                 span_distance_bin: int,
                                 head_scorer: TimeDistributed) -> torch.FloatTensor:
    """
    Computes an embedded representation of every candidate span. This is a concatenation
    of the contextualized endpoints of the span, an embedded representation of the width of
    the span and a representation of the span's predicted head. Also contains a bunch of features
    with respect to the target.

    Parameters
    ----------
    encoded_text : ``torch.FloatTensor``, required.
        The deeply embedded sentence of shape (batch_size, sequence_length, embedding_dim)
        over which we are computing a weighted sum.
    span_starts : ``torch.IntTensor``, required.
        A tensor of shape (batch_size, num_spans) representing the start of each span candidate.
    span_ends : ``torch.IntTensor``, required.
        A tensor of shape (batch_size, num_spans) representing the end of each span candidate.
    Returns
    -------
    span_embeddings : ``torch.FloatTensor``
        An embedded representation of every candidate span with shape:
        (batch_size, sentence_length, span_width, context_layer.get_output_dim() * 2 + embedding_size + feature_size)
    """
    # Shape: (batch_size, sequence_length, encoding_dim)
    # TODO(Swabha): necessary to have this? is it going to mess with attention computation?
    # contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
    _, sequence_length, _ = encoded_text.size()
    contextualized_embeddings = encoded_text

    # Shape: (batch_size, num_spans, encoding_dim)
    batch_size, num_spans = span_starts.size()
    assert num_spans == sequence_length * max_span_width

    start_embeddings = util.batched_index_select(
        contextualized_embeddings, span_starts.squeeze(-1))
    end_embeddings = util.batched_index_select(
        contextualized_embeddings, span_ends.squeeze(-1))

    # Compute and embed the span_widths (strictly speaking the span_widths - 1)
    # Shape: (batch_size, num_spans, 1)
    span_widths = span_ends - span_starts
    # Shape: (batch_size, num_spans, encoding_dim)
    span_width_embeddings = span_width_embedding(
        span_widths.squeeze(-1))

    target_index = target_index.view(batch_size, 1)
    span_dist = torch.abs(span_ends - target_index)
    span_dist = span_dist * (span_dist < span_distance_bin).long()
    span_dist_embeddings = span_distance_embedding(
        span_dist.squeeze(-1))

    span_dir = ((span_ends - target_index) > 0).long()
    span_dir_embeddings = span_direction_embedding(span_dir.squeeze(-1))

    # Shape: (batch_size, sequence_length, 1)
    head_scores = head_scorer(contextualized_embeddings)

    # Shape: (batch_size, num_spans, embedding_dim)
    # Note that we used the original text embeddings, not the contextual ones here.
    attended_text_embeddings = create_attended_span_representations(max_span_width,
                                                                    head_scores,
                                                                    encoded_text,
                                                                    span_ends,
                                                                    span_widths)
    # (batch_size, num_spans, context_layer.get_output_dim() * 3 + 2 * feature_dim)
    span_embeddings = torch.cat([start_embeddings,
                                 end_embeddings,
                                 span_width_embeddings,
                                 span_dist_embeddings,
                                 span_dir_embeddings,
                                 attended_text_embeddings], -1)
    span_embeddings = span_embeddings.view(
        batch_size, sequence_length, max_span_width, -1)
    return span_embeddings


def compute_simple_span_representations(max_span_width: int,
                                        encoded_text: torch.FloatTensor,
                                        span_starts: torch.IntTensor,
                                        span_ends: torch.IntTensor,
                                        span_width_embedding: Embedding,
                                        head_scorer: TimeDistributed) -> torch.FloatTensor:
    """
    Computes an embedded representation of every candidate span. This is a concatenation
    of the contextualized endpoints of the span, an embedded representation of the width of
    the span and a representation of the span's predicted head.

    Parameters
    ----------
    encoded_text : ``torch.FloatTensor``, required.
        The deeply embedded sentence of shape (batch_size, sequence_length, embedding_dim)
        over which we are computing a weighted sum.
    span_starts : ``torch.IntTensor``, required.
        A tensor of shape (batch_size, num_spans) representing the start of each span candidate.
    span_ends : ``torch.IntTensor``, required.
        A tensor of shape (batch_size, num_spans) representing the end of each span candidate.
    Returns
    -------
    span_embeddings : ``torch.FloatTensor``
        An embedded representation of every candidate span with shape:
        (batch_size, sentence_length, span_width, context_layer.get_output_dim() * 2 + embedding_size + feature_size)
    """
    # Shape: (batch_size, sequence_length, encoding_dim)
    # TODO(Swabha): necessary to have this? is it going to mess with attention computation?
    # contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
    _, sequence_length, _ = encoded_text.size()
    contextualized_embeddings = encoded_text

    # Shape: (batch_size, num_spans, encoding_dim)
    batch_size, num_spans = span_starts.size()
    assert num_spans == sequence_length * max_span_width

    start_embeddings = util.batched_index_select(
        contextualized_embeddings, span_starts.squeeze(-1))
    end_embeddings = util.batched_index_select(
        contextualized_embeddings, span_ends.squeeze(-1))

    # Compute and embed the span_widths (strictly speaking the span_widths - 1)
    # Shape: (batch_size, num_spans, 1)
    span_widths = span_ends - span_starts
    # Shape: (batch_size, num_spans, encoding_dim)
    span_width_embeddings = span_width_embedding(
        span_widths.squeeze(-1))

    # Shape: (batch_size, sequence_length, 1)
    head_scores = head_scorer(contextualized_embeddings)

    # Shape: (batch_size, num_spans, embedding_dim)
    # Note that we used the original text embeddings, not the contextual ones here.
    attended_text_embeddings = create_attended_span_representations(max_span_width,
                                                                    head_scores,
                                                                    encoded_text,
                                                                    span_ends,
                                                                    span_widths)
    # (batch_size, num_spans, context_layer.get_output_dim() * 3 + 2 * feature_dim)
    span_embeddings = torch.cat([start_embeddings,
                                 end_embeddings,
                                 span_width_embeddings,
                                 attended_text_embeddings], -1)
    span_embeddings = span_embeddings.view(
        batch_size, sequence_length, max_span_width, -1)
    return span_embeddings


def create_attended_span_representations(max_span_width: int,
                                         head_scores: torch.FloatTensor,
                                         encoded_text: torch.FloatTensor,
                                         span_ends: torch.IntTensor,
                                         span_widths: torch.IntTensor) -> torch.FloatTensor:
    """
    Given a tensor of unnormalized attention scores for each word in the document, compute
    distributions over every span with respect to these scores by normalising the headedness
    scores for words inside the span.

    Given these headedness distributions over every span, weight the corresponding vector
    representations of the words in the span by this distribution, returning a weighted
    representation of each span.

    Parameters
    ----------
    head_scores : ``torch.FloatTensor``, required.
        Unnormalized headedness scores for every word. This score is shared for every
        candidate. The only way in which the headedness scores differ over different
        spans is in the set of words over which they are normalized.
    text_embeddings: ``torch.FloatTensor``, required.
        The embeddings with shape  (batch_size, document_length, embedding_size)
        over which we are computing a weighted sum.
    span_ends: ``torch.IntTensor``, required.
        A tensor of shape (batch_size, num_spans), representing the end indices
        of each span.
    span_widths : ``torch.IntTensor``, required.
        A tensor of shape (batch_size, num_spans) representing the width of each
        span candidates.
    Returns
    -------
    attended_text_embeddings : ``torch.FloatTensor``
        A tensor of shape (batch_size, num_spans, embedding_dim) - the result of
        applying attention over all words within each candidate span.
    """
    # Shape: (1, 1, max_span_width)
    max_span_range_indices = util.get_range_vector(max_span_width,
                                                   encoded_text.is_cuda).view(1, 1, -1)

    # Shape: (batch_size, num_spans, max_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.

    span_ends = span_ends.unsqueeze(-1)
    span_widths = span_widths.unsqueeze(-1)
    span_mask = (max_span_range_indices <= span_widths).float()
    raw_span_indices = span_ends - max_span_range_indices
    # We also don't want to include span indices which are less than zero,
    # which happens because some spans near the beginning of the document
    # are of a smaller width than max_span_width, so we add this to the mask here.
    span_mask = span_mask * (raw_span_indices >= 0).float()
    # Spans
    span_indices = F.relu(raw_span_indices.float()).long()

    # Shape: (batch_size * num_spans * max_span_width)
    flat_span_indices = util.flatten_and_batch_shift_indices(
        span_indices, encoded_text.size(1))

    # Shape: (batch_size, num_spans, max_span_width, embedding_dim)
    span_text_embeddings = util.batched_index_select(
        encoded_text, span_indices, flat_span_indices)

    # Shape: (batch_size, num_spans, max_span_width)
    span_head_scores = util.batched_index_select(
        head_scores, span_indices, flat_span_indices).squeeze(-1)

    # Shape: (batch_size, num_spans, max_span_width)
    span_head_weights = util.last_dim_softmax(span_head_scores, span_mask)

    # Do a weighted sum of the embedded spans with
    # respect to the normalised head score distributions.
    # Shape: (batch_size, num_spans, embedding_dim)
    attended_text_embeddings = util.weighted_sum(
        span_text_embeddings, span_head_weights)

    return attended_text_embeddings


def get_tag_mask(num_classes: int, valid_frame_elements: Variable, batch_size: int) -> Variable:
    """
    Given a map of valid frame elements per instance, creates a mask for tags.
    """
    # Intended tag mask shape.
    zeros = torch.zeros(batch_size, num_classes).cuda()
    valid_frame_elements = valid_frame_elements.view(batch_size, -1)
    indices = F.relu(valid_frame_elements.float()
                     ).long().view(batch_size, -1).data
    values = (valid_frame_elements >= 0).data.float()
    tag_mask = zeros.scatter_(1, indices, values)
    return Variable(tag_mask)


def write_to_conll_eval_file(prediction_file: TextIO,
                             gold_file: TextIO,
                             verb_index: Optional[int],
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str]):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    verb_only_sentence = ["-"] * len(sentence)
    if verb_index:
        verb_only_sentence[verb_index] = sentence[verb_index]

    conll_format_predictions = convert_bio_tags_to_conll_format(prediction)
    conll_format_gold_labels = convert_bio_tags_to_conll_format(gold_labels)

    for word, predicted, gold in zip(verb_only_sentence,
                                     conll_format_predictions,
                                     conll_format_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")


def convert_bio_tags_to_conll_format(labels: List[str]):
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").

    A full example of the conversion performed:

    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]

    Parameters
    ----------
    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.

    Returns
    -------
    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels
