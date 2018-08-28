# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers.framenet.full_text_reader import FrameNetFullTextReader
from allennlp.data.dataset_readers.framenet.ontology_reader import FrameOntology
from allennlp.common.testing import AllenNlpTestCase


class TestFullTextReader(AllenNlpTestCase):

    def test_read_from_file(self):
        max_span_width = 10
        reader = FrameNetFullTextReader(max_span_width=max_span_width,
                                        data_path="data/fndata-1.5/")
        dataset = reader.read('tests/fixtures/framenet/annotations/fulltext/')
        instances = dataset.instances

        # First Instance
        fields = instances[0].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["The", "magistrate", "set", "a", "preliminary", "hearing", "for",
                          "next", "Tuesday", "and", "ordered", "Pickett", "held", "without", "bond", "."]
        assert fields["targets"].labels == [0, 0, 0, 0,
                                            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        span_starts = [
            j-d if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        assert span_starts == [
            x.sequence_index for x in fields["span_starts"].field_list]
        span_ends = [
            j for j in range(len(tokens)) for d in range(max_span_width)]
        assert span_ends == [
            x.sequence_index for x in fields["span_ends"].field_list]
        # span_mask = [
        #     1 if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        # assert fields["span_mask"].labels == span_mask
        assert fields["frame"].tokens[0].text == "Calendric_unit"
        # assert fields["lu"].tokens[0].text == "Tuesday.n"
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[6][6] = "O"
        gold_spans[7][0] = "Relative_time"
        gold_spans[8][0] = "Unit"
        gold_spans[15][6] = "O"
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        expected_valid_frame_elements = ["Relative_time", "Name",
                                         "Whole", "Count", "Unit", "Salient_event", "O"]
        assert expected_valid_frame_elements == [
            x.label for x in fields["valid_frame_elements"].field_list]

        # Second Instance
        fields = instances[1].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["The", "magistrate", "set", "a", "preliminary", "hearing", "for",
                          "next", "Tuesday", "and", "ordered", "Pickett", "held", "without", "bond", "."]
        assert fields["targets"].labels == [0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        assert fields["frame"].tokens[0].text == "Bail_decision"
        # assert fields["lu"].tokens[0].text == "order.v"
        expected_valid_frame_elements = [
            "Accused", "Judge", "Status", "Time", "Place", "Manner", "Means", "Purpose", "O"]
        assert expected_valid_frame_elements == [
            x.label for x in fields["valid_frame_elements"].field_list]
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[1][1] = "Judge"
        gold_spans[10][8] = "O"
        gold_spans[11][0] = "Accused"
        gold_spans[14][2] = "Status"
        gold_spans[15][0] = "O"
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        assert span_starts == [
            x.sequence_index for x in fields["span_starts"].field_list]
        assert span_ends == [
            x.sequence_index for x in fields["span_ends"].field_list]
        # assert fields["span_mask"].labels == span_mask

        # Third Instance - Too long an annotation
        fields = instances[2].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["The", "magistrate", "set", "a", "preliminary", "hearing", "for",
                          "next", "Tuesday", "and", "ordered", "Pickett", "held", "without", "bond", "."]
        assert fields["targets"].labels == [0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        assert fields["frame"].tokens[0].text == "Bail_decision"
        # assert fields["lu"].tokens[0].text == "bond.n"
        expected_valid_frame_elements = [
            "Accused", "Judge", "Status", "Time", "Place", "Manner", "Means", "Purpose", "O"]
        assert expected_valid_frame_elements == [
            x.label for x in fields["valid_frame_elements"].field_list]
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[0][0] = "O"
        gold_spans[10][9] = "O"
        gold_spans[11][0] = "Accused"
        gold_spans[15][3] = "O"
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        assert span_starts == [
            x.sequence_index for x in fields["span_starts"].field_list]
        assert span_ends == [
            x.sequence_index for x in fields["span_ends"].field_list]
        # assert fields["span_mask"].labels == span_mask

        # Fourth Instance - FE annotations only at rank 2.
        fields = instances[3].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["The", "magistrate", "set", "a", "preliminary", "hearing", "for",
                          "next", "Tuesday", "and", "ordered", "Pickett", "held", "without", "bond", "."]
        assert fields["targets"].labels == [0, 0, 0, 0,
                                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert fields["frame"].tokens[0].text == "Preliminaries"
        # assert fields["lu"].tokens[0].text == "preliminary.a"
        expected_valid_frame_elements = ["Agent", "Preparatory_act", "Purpose", "Descriptor",
                                         "Degree", "Domain_of_relevance", "Time", "Place", "Preparatory_phase", "O"]
        assert expected_valid_frame_elements == [
            x.label for x in fields["valid_frame_elements"].field_list]
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[5][5] = "O"
        gold_spans[15][9] = "O"
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        assert span_starts == [
            x.sequence_index for x in fields["span_starts"].field_list]
        assert span_ends == [
            x.sequence_index for x in fields["span_ends"].field_list]
        # assert fields["span_mask"].labels == span_mask

        # Missing POS tag annotation and reordered to check if annotation is done right.
        # Fifth instance, no FE annotation.
        fields = instances[4].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["He", "did", "not", "enter", "a", "plea", "."]
        assert fields["targets"].labels == [0, 0, 0, 0, 0, 1, 0]
        assert fields["frame"].tokens[0].text == "Entering_of_plea"
        # assert fields["lu"].tokens[0].text == "plea.n"
        expected_valid_frame_elements = ["Judge", "Accused", "Charges", "Plea",
                                         "Time", "Place", "Court", "O"]
        assert expected_valid_frame_elements == [
            x.label for x in fields["valid_frame_elements"].field_list]
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[6][6] = "O"
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        span_starts = [
            j-d if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        assert span_starts == [
            x.sequence_index for x in fields["span_starts"].field_list]
        span_ends = [
            j for j in range(len(tokens)) for d in range(max_span_width)]
        assert span_ends == [
            x.sequence_index for x in fields["span_ends"].field_list]
        # span_mask = [
        #     1 if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        # assert fields["span_mask"].labels == span_mask

    def test_fill_missing_spans(self):
        sentence_length = 15
        fes = [(3, 3, "a"), (6, 10, "b"), (12, 12, "c")]
        reader = FrameNetFullTextReader(max_span_width=10,
                                        data_path="data/fndata-1.5/")
        refilled_fes = reader.fill_missing_spans(
            fes, sentence_length)
        expected_fes = [(0, 2, "O"), (3, 3, "a"), (4, 5, "O"),
                        (6, 10, "b"), (11, 11, "O"), (12, 12, "c"), (13, 14, "O")]
        assert refilled_fes == expected_fes

        fes = []
        sentence_length = 20
        refilled_fes = reader.fill_missing_spans(
            fes, sentence_length)
        expected_fes = [(0, 19, "O")]
        assert refilled_fes == expected_fes
