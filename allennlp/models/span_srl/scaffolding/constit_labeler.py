from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.span_srl import span_srl_util
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy


@Model.register("constit_labeler")
class ConstitLabeler(Model):
    """
    This ``ConstitLabeler`` simply encodes a span of text with a stacked ``Seq2SeqEncoder``,
    then predicts a constituency tag for each span in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    stacked_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    max_span_width: int, required
        Consider only the spans up to max_span_width. When used in a scaffolding experiment,
        make sure this is at least as long as the span width from the other task.
        TODO(Swabha): make rigorous somehow.
    span_width_features_size: int, required
        Dimensionality of the span width feature.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 span_feedforward: FeedForward,
                 max_span_width: int,
                 span_width_feature_size: int,
                 label_namespace: str = "labels",
                 embedding_dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(ConstitLabeler, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self.stacked_encoder = stacked_encoder

        if text_field_embedder.get_output_dim() != stacked_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the phrase_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            stacked_encoder.get_input_dim()))

        self.max_span_width = max_span_width
        self.span_width_embedding = Embedding(max_span_width, span_width_feature_size)

        self.span_feedforward = TimeDistributed(span_feedforward)
        self.head_scorer = TimeDistributed(
            torch.nn.Linear(stacked_encoder.get_output_dim(), 1))

        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        self.tag_projection_layer = TimeDistributed(Linear(span_feedforward.get_output_dim(),
                                                           self.num_classes))

        # self.span_metric = SpanBasedF1Measure(vocab, tag_namespace=label_namespace)
        # Using accuracy as the metric, span F1 is overkill.
        self.span_metric = {"accuracy": CategoricalAccuracy()}

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                target_index: torch.LongTensor,
                span_starts: torch.LongTensor,
                span_ends: torch.LongTensor,
                span_mask: torch.LongTensor,
                tags: torch.LongTensor = None,
                parent_tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        target_index:
            Not required in this model, but necessary for scaffolding.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.embedding_dropout(
            self.text_field_embedder(tokens))
        batch_size, _, _ = embedded_text_input.size()
        mask = util.get_text_field_mask(tokens)
        encoded_text = self.stacked_encoder(embedded_text_input, mask)

        span_starts = F.relu(span_starts.float()).long().view(batch_size, -1)
        span_ends = F.relu(span_ends.float()).long().view(batch_size, -1)
        num_spans = span_starts.size(1)
        # Shape (batch_size, sequence_length * max_span_width, feature_dim)
        span_embeddings = span_srl_util.compute_simple_span_representations(self.max_span_width,
                                                                            encoded_text,
                                                                            span_starts,
                                                                            span_ends,
                                                                            self.span_width_embedding,
                                                                            self.head_scorer)
        # Shape (batch_size, sequence_length * max_span_width, 1)
        span_scores = self.span_feedforward(
            span_embeddings.view(batch_size, num_spans, -1))
        # Shape (batch_size, sequence_length * max_span_width, self.num_classes)
        logits = self.tag_projection_layer(span_scores)

        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          -1,
                                                                          self.num_classes])

        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities}

        if tags is not None:
            tags = tags.view(batch_size, -1)  # Flattening it out.
            loss = util.sequence_cross_entropy_with_logits(
                logits, tags, span_mask)
            for metric_name in self.span_metric:
                self.span_metric[metric_name](class_probabilities,
                                              tags, span_mask.float())
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i]
                                for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in argmax_indices]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # metric_dict = self.span_metric.get_metric(reset=reset)
        # return {metric_name: metric for metric_name, metric in metric_dict.items() if "overall" in metric_name}
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.span_metric.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ConstitLabeler':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(
            params.pop("stacked_encoder"))
        span_feedforward = FeedForward.from_params(
            params.pop("span_feedforward"))
        max_span_width = params.pop("max_span_width")
        span_width_feature_size = params.pop("span_width_feature_size")

        initializer = InitializerApplicator.from_params(
            params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   span_feedforward=span_feedforward,
                   max_span_width=max_span_width,
                   span_width_feature_size=span_width_feature_size,
                   initializer=initializer,
                   regularizer=regularizer)
