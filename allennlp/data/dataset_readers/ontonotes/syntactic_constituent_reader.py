import codecs
from collections import defaultdict
import os
import logging
from typing import Dict, List, Optional, Tuple

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ListField, IndexField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("syntactic_constit")
class SyntacticConstitReader(DatasetReader):
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

    def __init__(self, max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_namespace: str = "labels",
                 parent_label_namespace: str = "parent_labels") -> None:
        self.max_span_width = max_span_width
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace
        self.parent_label_namespace = parent_label_namespace
        self._tag_widths: Dict[str, List[int]] = {}

    def _process_sentence(self,
                          sentence_tokens: List[str],
                          predicate_index: int,
                          constits: Dict[Tuple[int, int], str],
                          parents: Dict[Tuple[int, int], str]) -> Instance:
        """
        Parameters
        ----------
        sentence_tokens : ``List[str]``, required.
            The tokenised sentence.
        predicate_index : ``int``, required.
            Index of the last predicate in the sentence.
        constits : ``Dict[Tuple[int, int], str]]``, required.

        Returns
        -------
        An instance.
        """
        def construct_matrix(labels: Dict[Tuple[int, int], str]) -> List[List[str]]:
            default = "*"

            def get_new_label(original: str, newer: str):
                return newer if original == default else "{}|{}".format(newer, original)

            constit_matrix = [[default for _ in range(self.max_span_width)]
                              for _ in sentence_tokens]
            for span in labels:
                start, end = span
                diff = end - start

                # Ignore the constituents longer than given maximum span width.
                if diff >= self.max_span_width:
                    continue
                # while diff >= self.max_span_width:
                #     old_label = constit_matrix[end][self.max_span_width - 1]
                #     constit_matrix[end][self.max_span_width -
                #                         1] = get_new_label(old_label, constits[span])
                #     end = end - self.max_span_width
                #     diff = end - start
                constit_matrix[end][diff] = get_new_label(
                    constit_matrix[end][diff], labels[span])
            return constit_matrix

        predicates = [0 for _ in sentence_tokens]
        predicates[predicate_index] = 1
        return self.text_to_instance(sentence_tokens,
                                     predicates,
                                     predicate_index,
                                     construct_matrix(constits),
                                     construct_matrix(parents))

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        sentence: List[str] = []
        open_constits: List[Tuple[str, int]] = []
        constits: Dict[Tuple[int, int], str] = {}
        parent_constits: Dict[Tuple[int, int], str] = {}
        predicate_index = None  # index of last predicate in the sentence.

        logger.info(
            "Reading syntactic instances from dataset files at: %s", file_path)
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

                            # Conll format data begins and ends with lines containing a hash,
                            # which may or may not occur after an empty line. To deal with this
                            # we check if the sentence is empty or not and if it is, we just skip
                            # adding instances, because there aren't any to add.
                            if not sentence:
                                continue
                            if not predicate_index:
                                predicate_index = int(len(sentence)/2)
                            instances.append(
                                self._process_sentence(sentence, predicate_index, constits, parent_constits))
                            # Reset everything for the next sentence.
                            sentence = []
                            open_constits = []
                            constits = {}
                            parent_constits = {}
                            predicate_index = None
                            continue

                        conll_components = line.split()
                        word = conll_components[3]

                        sentence.append(word)
                        word_index = len(sentence) - 1

                        # Heuristic: last predicate in the sentence is probably the most representative one.
                        if "(V*)" in line:
                            predicate_index = word_index

                        syn_label = conll_components[5]

                        if syn_label == "*":
                            continue
                        if "(" in syn_label:
                            starts = syn_label.split("(")
                            for con in starts[1:]:
                                label = con.strip(")").strip("*")
                                open_constits.append((label, word_index))
                        if ")" in syn_label:
                            ends = syn_label.count(")")
                            for _ in range(ends):
                                assert open_constits
                                label, start = open_constits.pop()
                                parent_label = "*"
                                if open_constits:
                                    parent_label, _ = open_constits[-1]
                                if (start, word_index) in constits:
                                    label = "{}|{}".format(
                                        constits[(start, word_index)], label)
                                constits[(start, word_index)] = label
                                parent_constits[(
                                    start, word_index)] = "{}-{}".format(parent_label, label)
                                if label not in self._tag_widths:
                                    self._tag_widths[label] = []
                                self._tag_widths[label].append(
                                    word_index - start + 1)

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        logger.info("# instances = %d", len(instances))
        # self.analyze_span_width()
        return Dataset(instances)

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
            logger.info("recall loss at length %d = %f", x[-1][0], x[-1][1])
            # print(x[-1])

        logger.info("avg tag length = %d", (total_tag_width / total_spans))
        import ipdb
        ipdb.set_trace()

    def text_to_instance(self,  # type: ignore
                         sentence_tokens: List[str],
                         predicates: List[int],
                         predicate_index: int,
                         constits: List[List[str]] = None,
                         parents: List[List[str]] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        text_field = TextField(
            [Token(t) for t in sentence_tokens], token_indexers=self._token_indexers)
        verb_field = SequenceLabelField(predicates, text_field)
        predicate_field = IndexField(predicate_index, text_field)

        # Span-based output fields.
        span_starts: List[Field] = []
        span_ends: List[Field] = []
        span_mask: List[int] = [1 for _ in range(
            len(sentence_tokens) * self.max_span_width)]
        span_labels: Optional[List[str]] = [
        ] if constits is not None else None
        parent_labels: Optional[List[str]] = [
        ] if parents is not None else None

        for j in range(len(sentence_tokens)):
            for diff in range(self.max_span_width):
                width = diff
                if j - diff < 0:
                    # This is an invalid span.
                    span_mask[j * self.max_span_width + diff] = 0
                    width = j

                span_starts.append(IndexField(j - width, text_field))
                span_ends.append(IndexField(j, text_field))

                if constits is not None:
                    label = constits[j][diff]
                    span_labels.append(label)

                if parents is not None:
                    parent_labels.append(parents[j][diff])

        start_fields = ListField(span_starts)
        end_fields = ListField(span_ends)
        span_mask_fields = SequenceLabelField(span_mask, start_fields)

        fields: Dict[str, Field] = {"tokens": text_field,
                                    "targets": verb_field,
                                    "span_starts": start_fields,
                                    "span_ends": end_fields,
                                    "span_mask": span_mask_fields,
                                    "target_index": predicate_field}

        if constits:
            fields['tags'] = SequenceLabelField(span_labels,
                                                start_fields,
                                                label_namespace=self.label_namespace)
            fields['parent_tags'] = SequenceLabelField(parent_labels,
                                                       start_fields,
                                                       label_namespace=self.parent_label_namespace)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'PhraseSyntaxReader':
        max_span_width = params.pop("max_span_width")
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))
        label_namespace = params.pop("label_namespace", "labels")
        parent_label_namespace = params.pop(
            "parent_label_namespace", "parent_labels")
        params.assert_empty(cls.__name__)
        return SyntacticConstitReader(max_span_width=max_span_width,
                                      token_indexers=token_indexers,
                                      label_namespace=label_namespace,
                                      parent_label_namespace=parent_label_namespace)
