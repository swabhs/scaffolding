import codecs
import os
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import tqdm
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ListField, SequenceLabelField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("span_annotation")
class SpanAnnotationReader(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    in the format used by the CoNLL 2011/2012 shared tasks. In order to use this
    Reader, you must follow the instructions provided `here (v12 release):
    <http://cemantix.org/data/ontonotes.html>`_, which will allow you to download
    the CoNLL style annotations for the  OntoNotes v5.0 release -- LDC2013T19.tgz
    obtained from LDC.

    Once you have run the scripts on the extracted data, you will have a folder
    structured as follows:

    conll-formatted-ontonotes-5.0/
     ── data
       ├── development
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       ├── test
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       └── train
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb

    The file path provided to this class can then be any of the train, test or development
    directories(or the top level data directory, if you are not utilizing the splits).

    The data has the following format, ordered by column.

    1 Document ID : str
        This is a variation on the document filename
    2 Part number : int
        Some files are divided into multiple parts numbered as 000, 001, 002, ... etc.
    3 Word number : int
        This is the word index of the word in that sentence.
    4 Word : str
        This is the token as segmented/tokenized in the Treebank. Initially the ``*_skel`` file
        contain the placeholder [WORD] which gets replaced by the actual token from the
        Treebank which is part of the OntoNotes release.
    5 POS Tag : str
        This is the Penn Treebank style part of speech. When parse information is missing,
        all part of speeches except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    6 Parse bit: str
        This is the bracketed structure broken before the first open parenthesis in the parse,
        and the word/part-of-speech leaf replaced with a ``*``. The full parse can be created by
        substituting the asterisk with the "([pos] [word])" string (or leaf) and concatenating
        the items in the rows of that column. When the parse information is missing, the
        first word of a sentence is tagged as ``(TOP*`` and the last word is tagged as ``*)``
        and all intermediate words are tagged with a ``*``.
    7 Predicate lemma: str
        The predicate lemma is mentioned for the rows for which we have semantic role
        information or word sense information. All other rows are marked with a "-".
    8 Predicate Frameset ID: int
        The PropBank frameset ID of the predicate in Column 7.
    9 Word sense: float
        This is the word sense of the word in Column 3.
    10 Speaker/Author: str
        This is the speaker or author name where available. Mostly in Broadcast Conversation
        and Web Log data. When not available the rows are marked with an "-".
    11 Named Entities: str
        These columns identifies the spans representing various named entities. For documents
        which do not have named entity annotation, each line is represented with an ``*``.
    12+ Predicate Arguments: str
        There is one column each of predicate argument structure information for the predicate
        mentioned in Column 7. If there are no predicates tagged in a sentence this is a
        single column with all rows marked with an ``*``.
    -1 Co-reference: str
        Co-reference chain information encoded in a parenthesis structure. For documents that do
         not have co-reference annotations, each line is represented with a "-".

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.
    """

    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self.max_span_width = max_span_width
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}

        # For analysis
        self._tag_widths: Dict[str, List[int]] = {}
        self._total_args = 0
        self._non_constit_args = 0
        self._constits_args = {}

    def calculate_span_size(self, tag_sequence: List[str]):
        def remove_bio(tag):
            return tag[2:] if tag.startswith('B-') or tag.startswith('I-') else tag
        sizes = [(remove_bio(tag_sequence[0]), 1)]

        for idx, tag in enumerate(tag_sequence[1:], 1):
            if tag != tag_sequence[idx-1] or tag.startswith("B-"):
                sizes.append((remove_bio(tag), 1))
            else:
                last_tag, length = sizes.pop()
                sizes.append((last_tag, length+1))

        for pair in sizes:
            tag, length = pair
            if tag not in self._tag_widths:
                self._tag_widths[tag] = []
            self._tag_widths[tag].append(length)

    def _convert_bio_into_matrix(self, tag_sequence: List[str]) -> List[List[str]]:
        def remove_bio(tag):
            return tag[2:] if tag.startswith('B-') or tag.startswith('I-') else tag
        # self.calculate_span_size(tag_sequence)

        spans = [["*" for _ in range(self.max_span_width)]
                 for _ in range(len(tag_sequence))]

        start_span = 0
        current_tag = tag_sequence[0]
        for pos, tag in enumerate(tag_sequence[1:], 1):
            width = pos - start_span
            if tag.startswith("B-") or (tag == "O" and tag_sequence[pos - 1] != "O"):
                width = pos - 1 - start_span
                spans[pos - 1][width] = remove_bio(current_tag)
                start_span = pos
                current_tag = tag
                width = pos - start_span
            elif width == self.max_span_width - 1:  # maximum allowed width
                spans[pos][width] = remove_bio(current_tag)
                start_span = pos + 1
                if pos + 1 < len(tag_sequence):
                    current_tag = tag_sequence[pos + 1]
        spans[len(tag_sequence)-1][len(tag_sequence)-1 -
                                   start_span] = remove_bio(tag_sequence[-1])
        return spans

    def _process_sentence(self,
                          sentence_tokens: List[str],
                          constits: Dict[Tuple[int, int], str],
                          verbal_predicates: List[int],
                          predicate_argument_labels: List[List[str]]) -> List[Instance]:
        """
        Parameters
        ----------
        sentence_tokens : ``List[str]``, required.
            The tokenised sentence.
        constits : ``Dict[Tuple[int, int], str]]``, required.
        verbal_predicates : ``List[int]``, required.
            The indexes of the verbal predicates in the
            sentence which have an associated annotation.
        predicate_argument_labels : ``List[List[str]]``, required.
            A list of predicate argument labels, one for each verbal_predicate. The
            internal lists are of length: len(sentence).

        Returns
        -------
        A list of Instances.

        """
        default = "*"

        def get_new_label(original: str, newer: str):
            return newer if original == default else "{}|{}".format(newer, original)

        constit_matrix = [[default for _ in range(self.max_span_width)]
                          for _ in sentence_tokens]
        for span in constits:
            start, end = span
            diff = end - start
            if diff >= self.max_span_width:
                continue
            # while diff >= self.max_span_width:
            #     old_label = constit_matrix[end][self.max_span_width - 1]
            #     constit_matrix[end][self.max_span_width -
            #                         1] = get_new_label(old_label, constits[span])
            #     end = end - self.max_span_width
            #     diff = end - start
            constit_matrix[end][diff] = get_new_label(
                constit_matrix[end][diff], constits[span])

        tokens = [Token(t) for t in sentence_tokens]
        if not verbal_predicates:
            # Sentence contains no predicates.
            tags = ["O" for _ in sentence_tokens]
            verb_label = [0 for _ in sentence_tokens]
            srl_args = self._convert_bio_into_matrix(tags)
            dummy_verb_index = 0
            return [self.text_to_instance(
                tokens, verb_label, dummy_verb_index, constit_matrix, srl_args)]
        else:
            instances = []

            for verb_index, tags in zip(verbal_predicates, predicate_argument_labels):
                verb_label = [0 for _ in sentence_tokens]
                verb_label[verb_index] = 1
                srl_args = self._convert_bio_into_matrix(tags)
                instances.append(self.text_to_instance(
                    tokens, verb_label, verb_index, constit_matrix, srl_args))
                self.find_overlap(srl_args, constit_matrix)
            return instances

    def find_overlap(self, srl_args, constit_matrix):
        for j in range(len(srl_args)):
            for diff in range(len(srl_args[0])):
                arg = srl_args[j][diff]
                constit = constit_matrix[j][diff]
                if arg == "*" or arg == "V" or arg == "O":
                    continue
                self._total_args += 1
                if constit == "*":
                    if j-diff != 0:
                        self._non_constit_args += 1
                    continue
                if constit not in self._constits_args:
                    self._constits_args[constit] = 0
                self._constits_args[constit] += 1

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        sentence: List[str] = []
        open_constits: List[Tuple[str, int]] = []
        constits: Dict[Tuple[int, int], str] = {}
        verbal_predicates: List[int] = []
        predicate_argument_labels: List[List[str]] = []
        current_span_label: List[Optional[str]] = []

        logger.info(
            "Reading phrase-syntax and SRL instances from dataset files at: %s", file_path)
        for root, _, files in tqdm.tqdm(list(os.walk(file_path))):
            for data_file in files:
                # These are a relic of the dataset pre-processing. Every file will be duplicated
                # - one file called filename.gold_skel and one generated from the preprocessing
                # called filename.gold_conll.
                if not data_file.endswith("gold_conll"):
                    continue
                with codecs.open(os.path.join(root, data_file), 'r', encoding='utf8') as open_file:
                    for line in open_file:
                        line = line.strip()
                        if line == '' or line.startswith("#"):

                            # CoNLL format data begins and ends with lines containing a hash,
                            # which may or may not occur after an empty line. To deal with this
                            # we check if the sentence is empty or not and if it is, we just skip
                            # adding instances, because there aren't any to add.
                            if not sentence:
                                continue
                            instances.extend(self._process_sentence(sentence,
                                                                    constits,
                                                                    verbal_predicates,
                                                                    predicate_argument_labels))
                            # Reset everything for the next sentence.
                            sentence = []
                            open_constits = []
                            constits = {}
                            verbal_predicates = []
                            predicate_argument_labels = []
                            current_span_label = []
                            continue

                        conll_components = line.split()
                        word = conll_components[3]

                        sentence.append(word)
                        word_index = len(sentence) - 1

                        # Extract phrase syntax.
                        syn_label = conll_components[5]
                        if syn_label != "*":
                            if "(" in syn_label:
                                starts = syn_label.split("(")
                                for con in starts[1:]:
                                    clabel = con.strip(")").strip("*")
                                    open_constits.append((clabel, word_index))
                            if ")" in syn_label:
                                ends = syn_label.count(")")
                                for _ in range(ends):
                                    assert open_constits
                                    clabel, start = open_constits.pop()
                                    if (start, word_index) in constits:
                                        clabel = "{}|{}".format(
                                            constits[(start, word_index)], clabel)
                                    constits[(start, word_index)] = clabel
                                    if clabel not in self._tag_widths:
                                        self._tag_widths[clabel] = []
                                    self._tag_widths[clabel].append(
                                        word_index - start + 1)

                        # Extract SRL arguments.
                        if word_index == 0:
                            # We're starting a new sentence. Here we set up a list of lists
                            # for the BIO labels for the annotation for each verb and create
                            # a temporary 'current_span_label' list for each annotation which
                            # we will use to keep track of whether we are beginning, inside of,
                            # or outside a particular span.
                            predicate_argument_labels = [
                                [] for _ in conll_components[11:-1]]
                            current_span_label = [
                                None for _ in conll_components[11:-1]]
                            prev_span_label = [
                                None for _ in conll_components[11:-1]]

                        num_annotations = len(predicate_argument_labels)
                        is_verbal_predicate = False

                        # Iterate over all verb annotations for the current sentence.
                        for annotation_index in range(num_annotations):
                            annotation = conll_components[11 +
                                                          annotation_index]

                            # If any annotation contains this word as a verb predicate,
                            # we need to record its index. This also has the side effect
                            # of ordering the verbal predicates by their location in the
                            # sentence, automatically aligning them with the annotations.
                            if "(V" in annotation:
                                is_verbal_predicate = True

                            label = annotation.strip("()*")

                            if "(" in annotation:
                                # Entering into a span for a particular semantic role label.
                                # We append the label and set the current span for this annotation.
                                bio_label = "B-" + label
                                predicate_argument_labels[annotation_index].append(
                                    bio_label)
                                current_span_label[annotation_index] = label
                            elif current_span_label[annotation_index] is not None:
                                # If there's no '(' token, but the current_span_label is not None,
                                # then we are inside a span.
                                bio_label = "I-" + \
                                    current_span_label[annotation_index]
                                predicate_argument_labels[annotation_index].append(
                                    bio_label)
                            else:
                                # We're outside a span.
                                predicate_argument_labels[annotation_index].append(
                                    "O")

                            prev_span_label[annotation_index] = current_span_label[annotation_index]

                            # Exiting a span, so we reset the current span label
                            # for this annotation.
                            if ")" in annotation:
                                current_span_label[annotation_index] = None

                        if is_verbal_predicate:
                            verbal_predicates.append(word_index)

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        logger.info("# instances = %d", len(instances))
        # self.analyze_span_width()
        # self.analyze_overlap()
        return Dataset(instances)

    def analyze_overlap(self):
        logger.info("%% non-constit SRL args = %f",
                    self._non_constit_args*100.0/self._total_args)
        from collections import OrderedDict
        from operator import itemgetter
        x = {k: self._constits_args[k] /
             self._total_args for k in self._constits_args}
        d = OrderedDict(sorted(x.items(),
                               key=itemgetter(1), reverse=True))
        print(d)

    def analyze_span_width(self):
        total_tag_width = 0.0
        total_spans = 0
        widths = defaultdict(int)

        for tag in self._tag_widths:
            if tag in ["*", "V", "O"]:
                continue
            total_tag_width += sum(self._tag_widths[tag])
            total_spans += len(self._tag_widths[tag])

            for l in self._tag_widths[tag]:
                widths[l] += 1

        x = []
        for l in sorted(widths):
            if len(x) == 0:
                x.append((l, widths[l]*1.0/total_spans))
            else:
                x.append((l, x[-1][1] + widths[l]*1.0/total_spans))
            print(x[-1])

        print("avg tag length = {}".format(total_tag_width / total_spans))
        import ipdb
        ipdb.set_trace()

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         verb_index: int,
                         constituents: List[List[str]] = None,
                         srl_args: List[List[str]] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ

        # Input fields.
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        verb_field = SequenceLabelField(verb_label, text_field)
        target_field = IndexField(verb_index, text_field)

        # Span-based output fields.
        span_starts: List[Field] = []
        span_ends: List[Field] = []
        span_mask: List[int] = [1 for _ in range(
            len(tokens) * self.max_span_width)]
        span_labels: Optional[List[str]] = [
        ] if srl_args is not None else None
        constit_labels: Optional[List[str]] = [
        ] if constituents is not None else None

        for j in range(len(tokens)):
            for diff in range(self.max_span_width):
                width = diff
                if j - diff < 0:
                    # This is an invalid span.
                    span_mask[j * self.max_span_width + diff] = 0
                    width = j

                span_starts.append(IndexField(j - width, text_field))
                span_ends.append(IndexField(j, text_field))

                if srl_args:
                    current_label = srl_args[j][diff]
                    span_labels.append(current_label)

                if constituents:
                    label = constituents[j][diff]
                    constit_labels.append(label)

        start_fields = ListField(span_starts)
        end_fields = ListField(span_ends)
        span_mask_fields = SequenceLabelField(span_mask, start_fields)

        fields: Dict[str, Field] = {'tokens': text_field,
                                    'verb_indicator': verb_field,
                                    'target_index': target_field,
                                    'span_starts': start_fields,
                                    'span_ends': end_fields,
                                    'span_mask': span_mask_fields}

        if srl_args:
            fields['tags'] = SequenceLabelField(span_labels, start_fields)
        if constituents:
            fields['constituents'] = SequenceLabelField(constit_labels,
                                                        start_fields,
                                                        label_namespace="constit_labels")
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'SpanAnnotationReader':
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))
        max_span_width = params.pop("max_span_width")
        params.assert_empty(cls.__name__)
        return SpanAnnotationReader(token_indexers=token_indexers,
                                    max_span_width=max_span_width)
