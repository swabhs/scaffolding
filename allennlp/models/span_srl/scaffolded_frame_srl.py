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
from allennlp.training.metrics import NonBioSpanBasedF1Measure


@Model.register("scaffolded_frame_srl")
class ScaffoldedFrameSrl(Model):
    """
    This model performs semantic role labeling for FrameNet.

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
    fast_mode: ``bool``, option (default=``True``)
        In this mode, we compute loss only on the train set and evaluate only on the dev set.
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
                 ontology_path: str,
                 embedding_dropout: float = 0.2,
                 srl_label_namespace: str = "labels",
                 constit_label_namespace: str = "constit_labels",
                 fast_mode: bool = True,
                 loss_type: str = "hamming",
                 unlabeled_constits: bool = False,
                 np_pp_constits: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(ScaffoldedFrameSrl, self).__init__(vocab, regularizer)

        # Base token-level encoding.
        self.text_field_embedder = text_field_embedder
        self.embedding_dropout = Dropout(p=embedding_dropout)
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.stacked_encoder = stacked_encoder
        if text_field_embedder.get_output_dim() + binary_feature_dim != stacked_encoder.get_input_dim():
            raise ConfigurationError("The input dimension of the stacked_encoder must be equal to "
                                     "the output dimension of the text_field_embedder.")

        # Span-level encoding.
        self.max_span_width = max_span_width
        self.span_width_embedding = Embedding(max_span_width,
                                              binary_feature_size)
        # Based on the average sentence length in FN train.
        self.span_distance_bin = 25
        self.span_distance_embedding = Embedding(self.span_distance_bin,
                                                 distance_feature_size)
        self.span_direction_embedding = Embedding(2, binary_feature_size)
        self.span_feedforward = TimeDistributed(span_feedforward)
        self.head_scorer = TimeDistributed(torch.nn.Linear(stacked_encoder.get_output_dim(),
                                                           1))

        self.num_srl_args = self.vocab.get_vocab_size(srl_label_namespace)
        self.not_a_span_tag = self.vocab.get_token_index("*", srl_label_namespace)
        self.outside_span_tag = self.vocab.get_token_index("O", srl_label_namespace)
        self.semi_crf = SemiMarkovConditionalRandomField(num_tags=self.num_srl_args,
                                                         max_span_width=max_span_width,
                                                         default_tag=self.not_a_span_tag,
                                                         outside_span_tag=self.outside_span_tag,
                                                         loss_type=loss_type)
        # self.crf = ConditionalRandomField(self.num_classes)
        self.unlabeled_constits = unlabeled_constits
        self.np_pp_constits = np_pp_constits
        self.constit_label_namespace = constit_label_namespace

        assert not (unlabeled_constits and np_pp_constits)
        if unlabeled_constits:
            self.num_constit_tags = 2
        elif np_pp_constits:
            self.num_constit_tags = 3
        else:
            self.num_constit_tags = self.vocab.get_vocab_size(
                constit_label_namespace)

        # Topmost MLP.
        self.srl_arg_projection_layer = TimeDistributed(
            Linear(span_feedforward.get_output_dim(), self.num_srl_args))
        self.constit_arg_projection_layer = TimeDistributed(
            Linear(span_feedforward.get_output_dim(), self.num_constit_tags))

        # Evaluation.
        self.metrics = {
            "constituents": NonBioSpanBasedF1Measure(vocab,
                                                     tag_namespace=constit_label_namespace,
                                                     ignore_classes=["*"]),
            "srl": NonBioSpanBasedF1Measure(vocab,
                                            tag_namespace=srl_label_namespace,
                                            ignore_classes=["O", "*"],
                                            ontology_path=ontology_path)
        }

        # Mode for the model, if turned on it only evaluates on dev and calculates loss for train.
        self.fast_mode = fast_mode
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                targets: torch.LongTensor,
                target_index: torch.LongTensor,
                span_starts: torch.LongTensor,
                span_ends: torch.LongTensor,
                tags: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
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
        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
        text_mask = util.get_text_field_mask(tokens)

        embedded_verb_indicator = self.binary_feature_embedding(targets.long())
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, embedded_verb_indicator], -1)
        embedding_dim_with_binary_feature = embedded_text_with_verb_indicator.size()[2]

        if self.stacked_encoder.get_input_dim() != embedding_dim_with_binary_feature:
            raise ConfigurationError("The SRL model uses an indicator feature, which makes "
                                     "the embedding dimension one larger than the value "
                                     "specified. Therefore, the 'input_dim' of the stacked_encoder "
                                     "must be equal to total_embedding_dim + 1.")

        encoded_text = self.stacked_encoder(embedded_text_with_verb_indicator, text_mask)

        batch_size, num_spans = tags.size()
        assert num_spans % self.max_span_width == 0
        tags = tags.view(batch_size, -1, self.max_span_width)

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

        # FN-specific parameters.
        fn_args = []
        for extra_arg in ['frame', 'valid_frame_elements']:
            if extra_arg in kwargs and kwargs[extra_arg] is not None:
                fn_args.append(kwargs[extra_arg])

        if fn_args:  # FrameSRL batch.
            frame, valid_frame_elements = fn_args
            output_dict = self.compute_srl_graph(span_scores=span_scores,
                                                 frame=frame,
                                                 valid_frame_elements=valid_frame_elements,
                                                 tags=tags,
                                                 text_mask=text_mask,
                                                 target_index=target_index)
        else:  # Scaffold batch.
            if "span_mask" in kwargs and kwargs["span_mask"] is not None:
                span_mask = kwargs["span_mask"]
            if "parent_tags" in kwargs and kwargs["parent_tags"] is not None:
                parent_tags = kwargs["parent_tags"]
            if self.unlabeled_constits:
                not_a_constit = self.vocab.get_token_index("*", self.constit_label_namespace)
                tags = (tags != not_a_constit).float().view(batch_size, -1, self.max_span_width)
            elif self.constit_label_namespace == "parent_labels":
                tags = parent_tags.view(batch_size, -1, self.max_span_width)
            elif self.np_pp_constits:
                tags = self.get_new_tags_np_pp(tags, batch_size)
            output_dict = self.compute_constit_graph(span_mask=span_mask,
                                                     span_scores=span_scores,
                                                     constit_tags=tags,
                                                     text_mask=text_mask)

        if self.fast_mode and not self.training:
            output_dict["loss"] = Variable(torch.FloatTensor([0.00]))

        return output_dict

    def compute_srl_graph(self, span_scores, frame, valid_frame_elements, tags, text_mask, target_index):
        srl_logits = self.srl_arg_projection_layer(span_scores)
        output_dict = {"mask": text_mask, "srl_logits": srl_logits}

        batch_size = tags.size(0)
        # Mask out the invalid frame-elements.
        tag_mask = span_srl_util.get_tag_mask(
            self.num_srl_args, valid_frame_elements, batch_size)

        # Viterbi decoding
        if not self.training or (self.training and not self.fast_mode):
            srl_prediction, srl_probabilites = self.semi_crf.viterbi_tags(srl_logits,
                                                                           text_mask,
                                                                           tag_masks=tag_mask)
            output_dict["tags"] = srl_prediction
            output_dict["srl_probabilities"] = srl_probabilites

            frames = [self.vocab.get_token_from_index(f[0], "frames") for f in frame["frames"].data.tolist()]
            srl_prediction = srl_prediction.view(batch_size, -1, self.max_span_width)
            self.metrics["srl"](predictions=srl_prediction,
                                gold_labels=tags,
                                mask=text_mask,
                                frames=frames,
                                target_indices=target_index)

        # Loss computation
        if self.training or (not self.training and not self.fast_mode):
            if tags is not None:
                srl_log_likelihood, _ = self.semi_crf(srl_logits,
                                                      tags,
                                                      text_mask,
                                                      tag_mask=tag_mask)
                output_dict["loss"] = -srl_log_likelihood

        return output_dict

    def compute_constit_graph(self, span_scores, span_mask, constit_tags, text_mask):
        batch_size = text_mask.size(0)
        # Shape (batch_size, sequence_length * max_span_width, self.num_classes)
        constit_logits = self.constit_arg_projection_layer(span_scores)
        output_dict = {"mask": text_mask, "constit_logits": constit_logits}

        # Decoding
        if not self.training or (self.training and not self.fast_mode):
            reshaped_log_probs = constit_logits.view(-1, self.num_constit_tags)
            constit_probabilities = F.softmax(reshaped_log_probs,                                                                                  dim=-1).view(batch_size, -1, self.num_constit_tags)
            constit_predictions = constit_probabilities.max(-1)[1]
            output_dict["constit_probabilities"] = constit_probabilities
            self.metrics["constituents"](predictions=constit_predictions.view(batch_size, -1, self.max_span_width),
                                         gold_labels=constit_tags,
                                         mask=text_mask)

        # Loss computation
        if self.training or (not self.training and not self.fast_mode):
            if constit_tags is not None:
                # Flattening it out.
                flat_tags = constit_tags.view(batch_size, -1)
                cross_entropy_loss = util.sequence_cross_entropy_with_logits(constit_logits,
                                                                             flat_tags,
                                                                             span_mask)
                output_dict["loss"] = cross_entropy_loss

        return output_dict

    def get_new_tags_np_pp(self, tags: torch.Tensor, batch_size: int)-> torch.Tensor:
        not_a_constit = self.vocab.get_token_index(
            "*", self.constit_label_namespace)
        np_constit = self.vocab.get_token_index(
            "NP", self.constit_label_namespace)
        pp_constit = self.vocab.get_token_index(
            "PP", self.constit_label_namespace)

        other_tags = (tags != not_a_constit) & (tags != np_constit) & (tags != pp_constit)
        np_pp_tags = (tags == np_constit) | (tags == pp_constit)
        non_constit_tags = (tags == not_a_constit)
        all_tags = 0 * non_constit_tags + 1 * np_pp_tags + 2 * other_tags
        return all_tags.float().view(batch_size, -1, self.max_span_width)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert predicted tag matrix into a sequence of tag string labels.
        """
        tag_matrices = output_dict["tags"]
        sequence_lengths = util.get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()
        tag_matrix_list = tag_matrices.cpu().tolist()
        # TODO(Swabha): add case for non-batched.

        all_tag_sequences = []
        for tag_matrix, sent_len in zip(tag_matrix_list, sequence_lengths):
            tag_sequence = self.convert_spans_into_sequence_of_tags(tag_matrix, sent_len)
            all_tag_sequences.append(tag_sequence)

        output_dict["tags"] = all_tag_sequences
        return output_dict

    def convert_spans_into_sequence_of_tags(self,
                                            tag_matrix: List[List[int]],
                                            sentence_length: int) -> List[int]:
        tag_sequence = [self.outside_span_tag for _ in range(sentence_length)]
        end_idx = 0
        for span_list in tag_matrix:
            if end_idx > sentence_length:
                break
            diff = 0
            assert len(span_list) == self.max_span_width
            for span_tag in span_list:
                if diff > end_idx:
                    break
                if span_tag == self.not_a_span_tag:
                    continue
                start_idx = end_idx - diff
                for position in range(start_idx, end_idx+1):
                    # Make sure that the current position is not already assigned.
                    assert tag_sequence[position] == self.outside_span_tag
                    tag_sequence[position] = span_tag
                diff += 1
            end_idx += 1
        return tag_sequence

    def get_metrics(self, reset: bool = False):
        # short = {"precision-overall": "c-prec",
        #          "recall-overall": "c-rec",
        #          "f1-measure-overall": "c-f1"}
        # metric_dict = {}
        # for task in self.metrics:
        #     task_metric_dict = self.metrics[task].get_metric(reset=reset)
        #     for x, y in task_metric_dict.items():
        #         if "overall" in x:
        #             if task == "constituents":
        #                 metric_dict[short[x]] = y
        #             else:
        #                 metric_dict[x] = y
        #     # if self.training:
        #     # This can be a lot of metrics, as there are 3 per class.
        #     # During training, we only really care about the overall
        #     # metrics, so we filter for them here.
        #     # TODO(Mark): This is fragile and should be replaced with some verbosity level in Trainer.
        metric_dict = self.metrics["srl"].get_metric(reset=reset)
        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ScaffoldedFrameSrl':
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
        ontology_path = params.pop("ontology_path")
        fast_mode = params.pop("fast_mode", True)
        loss_type = params.pop("loss_type", "hamming")
        srl_label_namespace = params.pop("label_namespace", "labels")
        constit_label_namespace = params.pop(
            "constit_label_namespace", "constit_labels")
        unlabeled_constits = params.pop('unlabeled_constits', False)
        np_pp_constits = params.pop('np_pp_constits', False)
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
                   ontology_path=ontology_path,
                   binary_feature_size=binary_feature_size,
                   distance_feature_size=distance_feature_size,
                   srl_label_namespace=srl_label_namespace,
                   constit_label_namespace=constit_label_namespace,
                   unlabeled_constits=unlabeled_constits,
                   np_pp_constits=np_pp_constits,
                   fast_mode=fast_mode,
                   loss_type=loss_type,
                   initializer=initializer,
                   regularizer=regularizer)
