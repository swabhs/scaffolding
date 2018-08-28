import codecs
import logging
import os
import xml.etree.ElementTree as ElementTree
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

import tqdm
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.framenet.ontology_reader import FrameOntology
from allennlp.data.fields import Field, IndexField, ListField, SequenceLabelField, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("fn_full")
class FrameNetFullTextReader(DatasetReader):
    """
    This DatasetReader is designed to read the FrameNet 1.x full text data in xml format.
    fndata-1.5/
    ├── development
    │   └── fulltext
    │       ├── ANC__110CYL072.xml
    │       ├── KBEval__MIT.xml
    │       ├── LUCorpus-v0.3__20000415_apw_eng-NEW.xml
    │       ├── LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml
    │       ├── Miscellaneous__Hijack.xml
    │       ├── NTI__NorthKorea_NuclearOverview.xml
    │       ├── NTI__WMDNews_062606.xml
    │       └── PropBank__TicketSplitting.xml
    ├── frame -> /homes/gws/swabha/acl2017/data/fndata-1.5/frame
    ├── frameIndex.xml -> /homes/gws/swabha/acl2017/data/fndata-1.5/frameIndex.xml
    ├── luIndex.xml -> /homes/gws/swabha/acl2017/data/fndata-1.5/luIndex.xml
    ├── test
    │   └── fulltext
    │       ├── ANC__110CYL067.xml
    │       ├── ANC__110CYL069.xml
    │       ├── ANC__112C-L013.xml
    │       ├── ANC__IntroHongKong.xml
    │       ├── ANC__StephanopoulosCrimes.xml
    │       ├── ANC__WhereToHongKong.xml
    │       ├── KBEval__atm.xml
    │       ├── KBEval__Brandeis.xml
    │       ├── KBEval__cycorp.xml
    │       ├── KBEval__parc.xml
    │       ├── KBEval__Stanford.xml
    │       ├── KBEval__utd-icsi.xml
    │       ├── LUCorpus-v0.3__20000410_nyt-NEW.xml
    │       ├── LUCorpus-v0.3__AFGP-2002-602187-Trans.xml
    │       ├── LUCorpus-v0.3__enron-thread-159550.xml
    │       ├── LUCorpus-v0.3__IZ-060316-01-Trans-1.xml
    │       ├── LUCorpus-v0.3__SNO-525.xml
    │       ├── LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml
    │       ├── Miscellaneous__Hound-Ch14.xml
    │       ├── Miscellaneous__SadatAssassination.xml
    │       ├── NTI__NorthKorea_Introduction.xml
    │       ├── NTI__Syria_NuclearOverview.xml
    │       └── PropBank__AetnaLifeAndCasualty.xml
    └── train
        ├── exemplar-train -> /homes/gws/swabha/acl2017/data/fndata-1.5/lu
        └── fulltext
            ├── ANC__110CYL068.xml
            ├── ANC__110CYL070.xml
            ├── ANC__110CYL200.xml
            ├── ANC__EntrepreneurAsMadonna.xml
            ├── ANC__HistoryOfGreece.xml
            ├── ANC__HistoryOfJerusalem.xml
            ├── ANC__HistoryOfLasVegas.xml
            ├── ANC__IntroJamaica.xml
            ├── ANC__IntroOfDublin.xml
            ├── C-4__C-4Text.xml
            ├── KBEval__lcch.xml
            ├── KBEval__LCC-M.xml
            ├── LUCorpus-v0.3__20000416_xin_eng-NEW.xml
            ├── LUCorpus-v0.3__20000419_apw_eng-NEW.xml
            ├── LUCorpus-v0.3__20000420_xin_eng-NEW.xml
            ├── LUCorpus-v0.3__20000424_nyt-NEW.xml
            ├── LUCorpus-v0.3__602CZL285-1.xml
            ├── LUCorpus-v0.3__AFGP-2002-600002-Trans.xml
            ├── LUCorpus-v0.3__AFGP-2002-600045-Trans.xml
            ├── LUCorpus-v0.3__artb_004_A1_E1_NEW.xml
            ├── LUCorpus-v0.3__artb_004_A1_E2_NEW.xml
            ├── LUCorpus-v0.3__CNN_AARONBROWN_ENG_20051101_215800.partial-NEW.xml
            ├── LUCorpus-v0.3__CNN_ENG_20030614_173123.4-NEW-1.xml
            ├── LUCorpus-v0.3__wsj_1640.mrg-NEW.xml
            ├── LUCorpus-v0.3__wsj_2465.xml
            ├── NTI__BWTutorial_chapter1.xml
            ├── NTI__ChinaOverview.xml
            ├── NTI__Iran_Biological.xml
            ├── NTI__Iran_Chemical.xml
            ├── NTI__Iran_Introduction.xml
            ├── NTI__Iran_Missile.xml
            ├── NTI__Iran_Nuclear.xml
            ├── NTI__Kazakhstan.xml
            ├── NTI__LibyaCountry1.xml
            ├── NTI__NorthKorea_ChemicalOverview.xml
            ├── NTI__NorthKorea_NuclearCapabilities.xml
            ├── NTI__Russia_Introduction.xml
            ├── NTI__SouthAfrica_Introduction.xml
            ├── NTI__Taiwan_Introduction.xml
            ├── NTI__WMDNews_042106.xml
            ├── NTI__workAdvances.xml
            ├── PropBank__BellRinging.xml
            ├── PropBank__ElectionVictory.xml
            ├── PropBank__LomaPrieta.xml
            ├── PropBank__PolemicProgressiveEducation.xml
            ├── QA__IranRelatedQuestions.xml
            └── SemAnno__Text1.xml
    """

    def __init__(self,
                 max_span_width: int,
                 data_path: str,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 frame_indexers: Dict[str, TokenIndexer] = None,
                 syn_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._max_span_width = max_span_width

        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._frame_indexers = frame_indexers or {"frames": SingleIdTokenIndexer()}
        self._syn_indexers = syn_indexers or {"syn_labels": SingleIdTokenIndexer()}

        self._ontology = FrameOntology(data_path)
        self._namespace = {"fn": "http://framenet.icsi.berkeley.edu"}
        self._tokenization_layers = ["BNC", "PENN"]

        self._num_sents = 0
        self._valid_sents = 0
        self._total_sentence_length = 0
        self._discontinuous_targets = 0
        self._adjacent_labeled_args = 0
        self._total_labeled_args = 0

    def _reset(self):
        self._num_sents = 0
        self._valid_sents = 0
        self._total_sentence_length = 0
        self._discontinuous_targets = 0
        self._adjacent_labeled_args = 0
        self._total_labeled_args = 0

    def fill_missing_spans(self, fe_list: List[Tuple[int, int, str]], sentence_length: int):
        """
        Also counts adjacent spans wiht same labels.
        """
        if not fe_list:
            return [(0, sentence_length-1, "O")]

        prev_label = None
        prev_end = -1
        args = sorted([x for x in fe_list])
        for arg in args:
            start, end, label = arg
            if prev_end != start - 1:
                fe_list.append((prev_end + 1, start-1, "O"))
            else:
                if prev_label == label:
                    self._adjacent_labeled_args += 1
            prev_end = end
            prev_label = label
        if prev_end != sentence_length-1:
            fe_list.append((prev_end+1, sentence_length-1, "O"))
        return sorted(fe_list)

    def _convert_frame_elements_to_matrix(self, fe_list: List[Tuple[int, int, str]], sentence_length: int) -> List[List[str]]:
        matrix = [["*" for _ in range(self._max_span_width)]
                  for _ in range(sentence_length)]
        self._total_labeled_args += len(fe_list)
        fe_map_refilled = self.fill_missing_spans(fe_list, sentence_length)

        for span in fe_map_refilled:
            start, end, label = span
            diff = end - start
            while diff >= self._max_span_width:
                matrix[end][self._max_span_width - 1] = label
                end = end - self._max_span_width
                diff = end - start
            matrix[end][diff] = label
        return matrix

    def _convert_labeled_spans_to_sequence(self, spans: List[Tuple[int, int, str]], sequence_len: int) -> List[str]:
        sequence = ["-"] * sequence_len
        for span in spans:
            start, end, label = span

            for position in range(start, end+1):
                assert sequence[position] == "-"
                sequence[position] = label
        return sequence

    def _process_sentence(self,
                          sentence_tokens: List[str],
                          targets: List[List[Tuple[int, int]]],
                          lexical_units: List[str],
                          frames: List[str],
                          frame_elements: List[List[Tuple[int, int, str]]],
                          phrase_types: List[List[Tuple[int, int, str]]],
                          grammar_functions: List[List[Tuple[int, int, str]]]) -> List[Instance]:
        instances = []
        for target, lex_unit, frame, fe, pt, gf in zip(targets, lexical_units, frames, frame_elements, phrase_types, grammar_functions):
            target_label = [0 for _ in sentence_tokens]
            if len(target) > 1:
                self._discontinuous_targets += 1
            for target_span in target:
                for idx in range(target_span[0], target_span[1] + 1):
                    assert target_label[idx] == 0
                    target_label[idx] = 1
            fe_matrix = self._convert_frame_elements_to_matrix(fe, len(sentence_tokens))

            # For simplicity, we assume the target index to be the first target position.
            target_index = target[0][0]

            pt_seq = self._convert_labeled_spans_to_sequence(pt, len(sentence_tokens))
            gf_seq = self._convert_labeled_spans_to_sequence(gf, len(sentence_tokens))
            instances.append(self.text_to_instance(
                sentence_tokens, target_label, target_index, lex_unit, frame, fe_matrix, pt_seq, gf_seq))
        self._valid_sents += 1
        self._total_sentence_length += len(sentence_tokens)
        return instances

    @overrides
    def read(self, file_path: str):
        logger.info(
            "Reading FrameNet full text instances from dataset files at: %s", file_path)

        instances = []
        # prev_len = len(instances)
        for root, _, directory in tqdm.tqdm(list(os.walk(file_path))):
            for data_file in sorted(directory):
                if not data_file.endswith(".xml"):
                    continue
                instances.extend(self.read_single_fulltext_file(
                    os.path.join(root, data_file)))
                # logger.info("%s: # instances = %d", data_file, len(instances) - prev_len)
                # prev_len = len(instances)
        logger.info("# instances = %d", len(instances))
        logger.info("# sentences = %d", self._num_sents)
        logger.info("# valid sentences = %d", self._valid_sents)
        logger.info("# avg tokens in sentence = %f",
                    self._total_sentence_length/self._valid_sents)
        logger.info("# discontinuous targets = %d",
                    self._discontinuous_targets)
        logger.info("%% adjacent spans with same label = %f (%d/%d)", self._adjacent_labeled_args /
                    self._total_labeled_args, self._adjacent_labeled_args, self._total_labeled_args)
        self._reset()
        return Dataset(instances)

    def read_single_fulltext_file(self, data_file: str):
        instances = []
        with codecs.open(data_file, "rb", "utf-8") as xml_file:
            tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        full_text_filename = data_file.split("/")[-1]
        is_test_file = "test" in data_file

        for sentence in root.findall("fn:sentence", self._namespace):
            tokens: List[str] = []
            pos_tags: List[str] = []
            starts: Dict[int, int] = {}
            ends: Dict[int, int] = {}
            targets: List[List[Tuple[int, int]]] = []
            lexical_units: List[str] = []
            frames: List[str] = []
            frame_elements: List[List[Tuple[int, int, str]]] = []
            phrase_types: List[List[Tuple[int, int, str]]] = []
            grammar_functs: List[List[Tuple[int, int, str]]] = []

            sentence_text = sentence.find("fn:text", self._namespace).text
            self._num_sents += 1

            annotations = sentence.findall("fn:annotationSet", self._namespace)
            # An annotation could either contain the tokenization for the sentence or a frame-semantic parse.
            for annotation in annotations:
                if "luName" in annotation.attrib and "frameName" in annotation.attrib:
                    # Ignore the unannotated instances in ONLY dev/train.
                    if annotation.attrib["status"] == "UNANN" and not is_test_file:
                        continue

                    # Get the LU, Frame and FEs for this sentence.
                    lex_unit = annotation.attrib["luName"]
                    frame = annotation.attrib["frameName"]
                    if frame == "Test35":  # Bogus frame.
                        continue

                    target_tokens = []
                    frame_element_list = []
                    phrase_type_list = []
                    grammar_funct_list = []
                    # Targets and frame-elements.
                    for layer in annotation.findall("fn:layer", self._namespace):
                        layer_type = layer.attrib["name"]
                        if layer_type == "Target":
                            # Recover the target span.
                            target_labels = layer.findall("fn:label", self._namespace)

                            # Some annotations have missing targets - ignore those.
                            if not target_labels:
                                logger.info("Skipping: Missing target label at %s in %s",
                                            annotation.attrib["ID"], full_text_filename)
                                break

                            # There can be discontinous targets.
                            for label in target_labels:
                                try:
                                    start_token = starts[int(label.attrib["start"])]
                                    end_token = ends[int(label.attrib["end"])]
                                except:
                                    logger.info("Skipping: Start/End labels missing for target annotation %s in %s",
                                                annotation.attrib["ID"], full_text_filename)
                                    continue
                                target_tokens.append((start_token, end_token))

                        elif layer.attrib["name"] == "FE" and layer.attrib["rank"] == "1":
                            # Recover the frame elements.
                            for label in layer.findall("fn:label", self._namespace):
                                if "itype" in label.attrib:
                                    continue
                                try:
                                    start_token = starts[int(label.attrib["start"])]
                                    end_token = ends[int(label.attrib["end"])]
                                    frame_element_list.append((start_token, end_token, label.attrib["name"]))
                                except:
                                    logger.info("Skipping: Frame-elements annotated for missing tokenization at annotation %s in %s",
                                                annotation.attrib["ID"], full_text_filename)
                                    continue
                        elif layer.attrib["name"] == "GF":
                            for label in layer.findall("fn:label", self._namespace):
                                try:
                                    start_token = starts[int(label.attrib["start"])]
                                    end_token = ends[int(label.attrib["end"])]
                                    grammar_funct_list.append((start_token, end_token, label.attrib["name"]))
                                except:
                                    continue
                        elif layer.attrib["name"] == "PT":
                            for label in layer.findall("fn:label", self._namespace):
                                try:
                                    start_token = starts[int(label.attrib["start"])]
                                    end_token = ends[int(label.attrib["end"])]
                                    phrase_type_list.append((start_token, end_token, label.attrib["name"]))
                                except:
                                    continue
                    if not target_tokens:
                        logger.info("Skipping: Missing target in annotation %s in %s", annotation.attrib["ID"], full_text_filename)
                        continue

                    if target_tokens in targets:
                        logger.info("Skipping: Repeated annotation %s for frame %s in %s", annotation.attrib["ID"], frame, full_text_filename)
                        continue
                    lexical_units.append(lex_unit)
                    frames.append(frame)
                    targets.append(target_tokens)
                    frame_elements.append(frame_element_list)
                    grammar_functs.append(grammar_funct_list)
                    phrase_types.append(phrase_type_list)
                else:
                    for layer in annotation.findall("fn:layer", self._namespace):
                        if layer.attrib["name"] not in self._tokenization_layers:
                            continue

                        tokenization = {}
                        for label in layer.findall("fn:label", self._namespace):
                            start = int(label.attrib["start"])
                            end = int(label.attrib["end"])
                            tokenization[(start, end)] = label.attrib["name"]

                        previous_end = -2
                        for start_end in sorted(tokenization):
                            start, end = start_end
                            if start != previous_end + 2:
                                logger.info(
                                    "Fixing: Missing tokenization at annotation %s in %s.", annotation.attrib["ID"], full_text_filename)
                                # Creating a new token.
                                dummy_start = previous_end + 2
                                dummy_end = start - 2
                                tokens.append(
                                    sentence_text[dummy_start: dummy_end + 1])
                                pos_tags.append("missing")
                                starts[dummy_start] = len(tokens) - 1
                                ends[dummy_end] = len(tokens) - 1

                            tokens.append(sentence_text[start: end + 1])
                            pos_tags.append(tokenization[start_end])
                            starts[start] = len(tokens) - 1
                            ends[end] = len(tokens) - 1
                            previous_end = end
                        break
            if not targets:
                # Sentence with missing target annotations, will be skipped.
                continue

            assert len(pos_tags) == len(tokens)
            assert len(targets) == len(lexical_units) == len(frames) == len(frame_elements)
            instances.extend(self._process_sentence(sentence_tokens=tokens,
                                                    targets=targets,
                                                    lexical_units=lexical_units,
                                                    frames=frames,
                                                    frame_elements=frame_elements,
                                                    phrase_types=phrase_types,
                                                    grammar_functions=grammar_functs))
        xml_file.close()
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(data_file))
        return instances

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         target_label: List[int],
                         target_index: int,
                         lex_label: str,
                         frame_label: str,
                         frame_elements: List[List[str]] = None,
                         phrase_types: List[str] = None,
                         grammar_functs: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        text_field = TextField([Token(t) for t in tokens], token_indexers=self._token_indexers)
        verb_field = SequenceLabelField(target_label, text_field)
        target_field = IndexField(target_index, text_field)

        frame_field = TextField([Token(frame_label)], token_indexers=self._frame_indexers)
        # lex_field = TextField([Token(lex_label)], token_indexers=self._token_indexers)
        phrase_type_field = TextField([Token(p) for p in phrase_types], token_indexers=self._syn_indexers)
        grammar_funct_field = TextField([Token(gf) for gf in grammar_functs], token_indexers=self._syn_indexers)

        # Span-based output fields.
        span_starts: List[Field] = []
        span_ends: List[Field] = []
        # span_mask: List[int] = [1 for _ in range(len(tokens) * self._max_span_width)]
        span_labels: Optional[List[str]] = [] if frame_elements is not None else None

        for j in range(len(tokens)):
            for diff in range(self._max_span_width):
                width = diff
                if j - diff < 0:
                    # This is an invalid span.
                    # span_mask[j * self._max_span_width + diff] = 0
                    width = j

                span_starts.append(IndexField(j - width, text_field))
                span_ends.append(IndexField(j, text_field))

                if frame_elements is not None:
                    current_label = frame_elements[j][diff]
                    span_labels.append(current_label)

        start_fields = ListField(span_starts)
        end_fields = ListField(span_ends)
        # span_mask_fields = SequenceLabelField(span_mask, start_fields)

        # Valid labels for frame.
        valid_frame_elements = ListField([LabelField(x) for x in self._ontology.frame_fe_map[frame_label] + ["O"]])

        fields: Dict[str, Field] = {'tokens': text_field,
                                    'targets': verb_field,
                                    'target_index': target_field,
                                    # 'lu': lex_field,
                                    'frame': frame_field,
                                    'span_starts': start_fields,
                                    'span_ends': end_fields,
                                    # 'span_mask': span_mask_fields,
                                    'valid_frame_elements': valid_frame_elements}
        if frame_elements:
            fields['tags'] = SequenceLabelField(span_labels, start_fields)
        if grammar_functs:
            fields['gf'] = grammar_funct_field
        if phrase_types:
            fields['pt'] = phrase_type_field
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'FrameNetFullTextReader':
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))
        frame_indexers = TokenIndexer.dict_from_params(
            params.pop('frame_indexers', {"frames": {"type": "single_id", "namespace": "frames"}}))
        syn_indexers = TokenIndexer.dict_from_params(
            params.pop('syn_indexers', {"syn_labels": {"type": "single_id", "namespace": "syn_labels"}}))
        max_span_width = params.pop('max_span_width')
        data_path = params.pop('data_path', None)
        params.assert_empty(cls.__name__)
        return cls(token_indexers=token_indexers,
                   frame_indexers=frame_indexers,
                   syn_indexers=syn_indexers,
                   max_span_width=max_span_width,
                   data_path=data_path)
