from typing import Dict, List, TextIO, Optional

import torch
from torch.autograd import Variable
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.span_srl import span_srl_util
from allennlp.modules import Seq2SeqEncoder, FeedForward, TimeDistributed, TextFieldEmbedder, SemiMarkovConditionalRandomField
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import NonBioSpanBasedF1Measure, SpanBasedF1Measure


@Model.register("pb_scaffold")
class PropBankScaffoldSpanSrl(Model):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it is an implmentation of `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the verbal predicate to generate predictions for in
    the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
    the predictions to contain valid BIO sequences.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    stacked_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    binary_feature_dim : int, required.
        The dimensionality of the embedding of the binary verb predicate features.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
        TODO(swabha) : may screw up?
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 span_feedforward: FeedForward,
                 binary_feature_dim: int,
                 max_span_width: int,
                 binary_feature_size: int,
                 distance_feature_size: int,
                 embedding_dropout: float = 0.2,
                 srl_label_namespace: str = "labels",
                 constit_label_namespace: str = "constit_labels",
                 mixing_ratio: float = 1.0,
                 cutoff_epoch: int = -1,
                 fast_mode: bool = True,
                 loss_type: str = "logloss",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(PropBankScaffoldSpanSrl, self).__init__(vocab, regularizer)

        # Base token-level encoding.
        self.text_field_embedder = text_field_embedder
        self.embedding_dropout = Dropout(p=embedding_dropout)
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.stacked_encoder = stacked_encoder
        if text_field_embedder.get_output_dim() + binary_feature_dim != stacked_encoder.get_input_dim():
            raise ConfigurationError("The SRL Model uses a binary verb indicator feature, meaning "
                                     "the input dimension of the stacked_encoder must be equal to "
                                     "the output dimension of the text_field_embedder + 1.")

        # Span-level encoding.
        self.max_span_width = max_span_width
        self.span_width_embedding = Embedding(max_span_width,
                                              binary_feature_size)
        # Based on the average sentence length in FN train. TODO(Swabha): find out for OntoNotes.
        self.span_distance_bin = 25
        self.span_distance_embedding = Embedding(self.span_distance_bin,
                                                 distance_feature_size)
        self.span_direction_embedding = Embedding(2, binary_feature_size)
        self.span_feedforward = TimeDistributed(span_feedforward)
        self.head_scorer = TimeDistributed(torch.nn.Linear(stacked_encoder.get_output_dim(),
                                                           1))

        self.num_srl_args = self.vocab.get_vocab_size(srl_label_namespace)
        not_a_span_tag = self.vocab.get_token_index("*", srl_label_namespace)
        outside_span_tag = self.vocab.get_token_index("O", srl_label_namespace)
        self.semi_crf = SemiMarkovConditionalRandomField(num_tags=self.num_srl_args,
                                                         max_span_width=max_span_width,
                                                         loss_type=loss_type,
                                                         default_tag=not_a_span_tag,
                                                         outside_span_tag=outside_span_tag)
        # self.crf = ConditionalRandomField(self.num_classes)
        self.num_constit_tags = self.vocab.get_vocab_size(
            constit_label_namespace)

        # Topmost MLP.
        self.srl_arg_projection_layer = TimeDistributed(
            Linear(span_feedforward.get_output_dim(), self.num_srl_args))
        self.constit_arg_projection_layer = TimeDistributed(
            Linear(span_feedforward.get_output_dim(), self.num_constit_tags))

        self.mixing_ratio = mixing_ratio
        self.cutoff_batch = cutoff_epoch
        self.batch = 0

        # Evaluation.
        self.metrics = {
            "constituents": NonBioSpanBasedF1Measure(vocab,
                                                     tag_namespace=constit_label_namespace,
                                                     ignore_classes=["*"]),
            "srl": NonBioSpanBasedF1Measure(vocab,
                                            tag_namespace=srl_label_namespace,
                                            ignore_classes=["V", "*"])
        }

        # Mode for the model, if turned on it only evaluates on dev and calculates loss for train.
        self.fast_mode = fast_mode
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                target_index: torch.LongTensor,
                span_starts: torch.LongTensor,
                span_ends: torch.LongTensor,
                span_mask: torch.LongTensor,
                constituents: torch.LongTensor = None,
                tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
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
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        bio : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        tags: shape ``(batch_size, num_spans)``
        span_starts: shape ``(batch_size, num_spans)``
        span_ends: shape ``(batch_size, num_spans)``

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
        self.batch += 1
        embedded_text_input = self.embedding_dropout(
            self.text_field_embedder(tokens))
        batch_size = embedded_text_input.size(0)
        text_mask = util.get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(
            verb_indicator.long())
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_verb_indicator = torch.cat(
            [embedded_text_input, embedded_verb_indicator], -1)
        embedding_dim_with_binary_feature = embedded_text_with_verb_indicator.size()[
            2]

        if self.stacked_encoder.get_input_dim() != embedding_dim_with_binary_feature:
            raise ConfigurationError("The SRL model uses an indicator feature, which makes "
                                     "the embedding dimension one larger than the value "
                                     "specified. Therefore, the 'input_dim' of the stacked_encoder "
                                     "must be equal to total_embedding_dim + 1.")
        encoded_text = self.stacked_encoder(embedded_text_with_verb_indicator,
                                            text_mask)

        span_starts = F.relu(span_starts.float()).long().view(batch_size, -1)
        span_ends = F.relu(span_ends.float()).long().view(batch_size, -1)
        target_index = F.relu(target_index.float()).long().view(batch_size)
        # shape (batch_size, sequence_length * max_span_width, embedding_dim)
        span_embeddings = span_srl_util.compute_span_representations(self.max_span_width,
                                                                     encoded_text,
                                                                     target_index,
                                                                     span_starts,
                                                                     span_ends,
                                                                     self.span_width_embedding,
                                                                     self.span_direction_embedding,
                                                                     self.span_distance_embedding,
                                                                     self.span_distance_bin,
                                                                     self.head_scorer)
        span_scores = self.span_feedforward(span_embeddings)

        srl_logits = self.srl_arg_projection_layer(span_scores)
        constit_logits = self.constit_arg_projection_layer(span_scores)
        output_dict = {"srl_logits": srl_logits,
                       "constit_logits": constit_logits,
                       "mask": text_mask}

        tags = tags.view(batch_size, -1, self.max_span_width)
        constituents = constituents.view(batch_size, -1, self.max_span_width)

        # Viterbi decoding
        if not self.training or (self.training and not self.fast_mode):
            srl_prediction, srl_probabilities = self.semi_crf.viterbi_tags(srl_logits,
                                                                           text_mask)
            output_dict["srl_tags"] = srl_prediction
            output_dict["srl_tag_probabilities"] = srl_probabilities
            self.metrics["srl"](predictions=srl_prediction.view(batch_size, -1, self.max_span_width),
                                gold_labels=tags,
                                mask=text_mask)

            reshaped_constit_logits = constit_logits.view(-1,
                                                          self.num_constit_tags)
            constit_probabilities = F.softmax(reshaped_constit_logits, dim=-1)
            constit_predictions = constit_probabilities.max(-1)[1]
            output_dict["constit_tags"] = constit_predictions
            output_dict["constit_probabilities"] = constit_probabilities

            constit_predictions = constit_predictions.view(batch_size,
                                                           -1,
                                                           self.max_span_width)
            self.metrics["constituents"](predictions=constit_predictions,
                                         gold_labels=constituents,
                                         mask=text_mask)

        # Loss computation
        if self.training or (not self.training and not self.fast_mode):
            if tags is not None:
                srl_log_likelihood, _ = self.semi_crf(srl_logits,
                                                      tags,
                                                      mask=text_mask)
                output_dict["srl_loss"] = -srl_log_likelihood
            if constituents is not None:
                # Flattening it out.
                constituents = constituents.view(batch_size, -1)
                constit_loss = util.sequence_cross_entropy_with_logits(constit_logits,
                                                                       constituents,
                                                                       span_mask)
                output_dict["constit_loss"] = constit_loss
            if tags is not None and constituents is not None:
                if self.batch > self.cutoff_batch:
                    output_dict["loss"] = - srl_log_likelihood + self.mixing_ratio * \
                        constit_loss
                else:
                    output_dict["loss"] = -srl_log_likelihood
        if self.fast_mode and not self.training:
            output_dict["loss"] = Variable(torch.FloatTensor([0.00]))

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Not necessary for us.
        """
        raise NotImplementedError

    def get_metrics(self, reset: bool=False):
        short = {"precision-overall": "c-prec",
                 "recall-overall": "c-rec",
                 "f1-measure-overall": "c-f1"}
        metric_dict = {}
        for task in self.metrics:
            task_metric_dict = self.metrics[task].get_metric(reset=reset)
            for x, y in task_metric_dict.items():
                if "overall" in x:
                    if task == "constituents":
                        metric_dict[short[x]] = y
                    else:
                        metric_dict[x] = y
            # if self.training:
            # This can be a lot of metrics, as there are 3 per class.
            # During training, we only really care about the overall
            # metrics, so we filter for them here.
            # TODO(Mark): This is fragile and should be replaced with some verbosity level in Trainer.
        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'PropBankScaffoldSpanSrl':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(
            params.pop("stacked_encoder"))
        span_feedforward = FeedForward.from_params(
            params.pop("span_feedforward"))
        binary_feature_dim = params.pop("binary_feature_dim")
        max_span_width = params.pop("max_span_width")
        binary_feature_size = params.pop("feature_size")
        distance_feature_size = params.pop("distance_feature_size", 5)
        fast_mode = params.pop("fast_mode", True)
        loss_type = params.pop("loss_type", "hamming")
        mixing_ratio = params.pop("mixing_ratio", 1.0)
        cutoff_epoch = params.pop("cutoff_epoch", -1)
        label_namespace = params.pop("label_namespace", "labels")
        initializer = InitializerApplicator.from_params(
            params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   binary_feature_dim=binary_feature_dim,
                   span_feedforward=span_feedforward,
                   max_span_width=max_span_width,
                   binary_feature_size=binary_feature_size,
                   distance_feature_size=distance_feature_size,
                   srl_label_namespace=label_namespace,
                   loss_type=loss_type,
                   mixing_ratio=mixing_ratio,
                   cutoff_epoch=cutoff_epoch,
                   fast_mode=fast_mode,
                   initializer=initializer,
                   regularizer=regularizer)
