# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers.ontonotes.span_annotation_reader import SpanAnnotationReader
from allennlp.common.testing import AllenNlpTestCase


class TestSpanAnnotationReader(AllenNlpTestCase):

    def check_spans(self, span_starts, span_ends, sent_len, max_span_width):
        for end_pos in range(sent_len):
            for diff in range(max_span_width):
                idx = end_pos * max_span_width + diff

                span_start = span_starts.field_list[idx].sequence_index
                if end_pos - diff >= 0:
                    expected_span_start = end_pos - diff
                    assert span_start == expected_span_start
                else:
                    expected_span_start = 0
                    assert span_start == expected_span_start

                span_end = span_ends.field_list[idx].sequence_index
                assert span_end == end_pos

    def test_read_from_file(self):
        max_span_width = 10
        reader = SpanAnnotationReader(max_span_width)
        dataset = reader.read('tests/fixtures/conll_2012/')
        instances = dataset.instances

        fields = instances[0].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["Mali", "government", "officials", "say", "the", "woman", "'s",
                          "confession", "was", "forced", "."]
        assert fields["verb_indicator"].labels[3] == 1
        # assert fields["bio"].labels == ['B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
        #                                 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[2][2] = 'ARG0'
        gold_spans[3][0] = 'V'
        gold_spans[9][5] = 'ARG1'
        gold_spans[10][0] = 'O'
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        self.check_spans(fields["span_starts"],
                         fields["span_ends"], len(tokens), max_span_width)

        gold_constit_spans = [['*' for d in range(max_span_width)]
                              for j in range(len(tokens))]
        gold_constit_spans[1][1] = 'NML'
        gold_constit_spans[2][2] = 'NP'
        gold_constit_spans[6][2] = 'NP'
        gold_constit_spans[7][3] = 'NP'
        gold_constit_spans[9][0] = 'ADJP'
        gold_constit_spans[9][1] = 'VP'
        gold_constit_spans[9][5] = 'S|SBAR'
        gold_constit_spans[9][6] = 'VP'
        # gold_spans[10][10] = 'S|TOP'
        gold_constit_spans_flat = [
            gs for rows in gold_constit_spans for gs in rows]
        assert fields["constituents"].labels == gold_constit_spans_flat
        span_mask = [
            1 if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        assert fields["span_mask"].labels == span_mask
        assert fields["target_index"].sequence_index == 3

        fields = instances[1].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["Mali", "government", "officials", "say", "the", "woman", "'s",
                          "confession", "was", "forced", "."]
        assert fields["verb_indicator"].labels[8] == 1
        # assert fields["bio"].labels == ['O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1',
        #                                 'I-ARG1', 'I-ARG1', 'B-V', 'B-ARG2', 'O']
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[9][0] = 'ARG2'
        gold_spans[8][0] = 'V'
        gold_spans[7][3] = 'ARG1'
        gold_spans[3][3] = 'O'
        gold_spans[10][0] = 'O'
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        self.check_spans(fields["span_starts"],
                         fields["span_ends"], len(tokens), max_span_width)
        assert fields["constituents"].labels == gold_constit_spans_flat
        assert fields["span_mask"].labels == span_mask
        assert fields["target_index"].sequence_index == 8

        fields = instances[2].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month', 'after',
                          'four', 'months', 'of', 'hearings', '.']
        assert fields["verb_indicator"].labels[2] == 1
        # assert fields["bio"].labels == ['B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARGM-TMP',
        #                                 'I-ARGM-TMP', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP',
        #                                 'I-ARGM-TMP', 'I-ARGM-TMP', 'O']
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[1][1] = 'ARG0'
        gold_spans[2][0] = 'V'
        gold_spans[4][1] = 'ARG1'
        gold_spans[6][1] = 'ARGM-TMP'
        gold_spans[11][4] = 'ARGM-TMP'
        gold_spans[12][0] = 'O'
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        self.check_spans(fields["span_starts"],
                         fields["span_ends"], len(tokens), max_span_width)
        gold_constit_spans = [['*' for d in range(max_span_width)]
                              for j in range(len(tokens))]
        gold_constit_spans[1][1] = 'NP'
        gold_constit_spans[4][1] = 'NP'
        gold_constit_spans[6][1] = 'NP'
        gold_constit_spans[9][1] = 'NP'
        gold_constit_spans[11][0] = 'NP'
        gold_constit_spans[11][1] = 'PP'
        gold_constit_spans[11][3] = 'NP'
        gold_constit_spans[11][4] = 'PP'
        gold_constit_spans[11][9] = 'VP'
        # gold_spans[12][12] = 'S|TOP'
        gold_constit_spans_flat = [
            gs for rows in gold_constit_spans for gs in rows]
        assert fields["constituents"].labels == gold_constit_spans_flat
        span_mask = [
            1 if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        assert fields["span_mask"].labels == span_mask
        assert fields["target_index"].sequence_index == 2

        fields = instances[3].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month', 'after',
                          'four', 'months', 'of', 'hearings', '.']
        assert fields["verb_indicator"].labels[11] == 1
        # assert fields["bio"].labels == ['O', 'O', 'O', 'O',
        #                                 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-V', 'O']
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[9][9] = 'O'
        gold_spans[10][0] = 'O'
        gold_spans[11][0] = 'V'
        gold_spans[12][0] = 'O'
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        self.check_spans(fields["span_starts"],
                         fields["span_ends"], len(tokens), max_span_width)
        assert fields["constituents"].labels == gold_constit_spans_flat
        assert fields["span_mask"].labels == span_mask
        assert fields["target_index"].sequence_index == 11

        # Tests a sentence with no verbal predicates.
        fields = instances[4].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["Denise", "Dillon", "Headline", "News", "."]
        assert fields["verb_indicator"].labels == [0, 0, 0, 0, 0]
        # assert fields["bio"].labels == ['O', 'O', 'O', 'O', 'O']
        gold_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_spans[4][4] = 'O'
        gold_spans_flat = [label for rows in gold_spans for label in rows]
        assert fields["tags"].labels == gold_spans_flat
        self.check_spans(fields["span_starts"],
                         fields["span_ends"], len(tokens), max_span_width)
        gold_constit_spans = [['*' for d in range(max_span_width)]
                              for j in range(len(tokens))]
        gold_constit_spans[1][1] = 'NP'
        gold_constit_spans[3][1] = 'NP'
        gold_constit_spans[4][4] = 'FRAG|TOP'
        gold_constit_spans_flat = [
            gs for rows in gold_constit_spans for gs in rows]
        assert fields["constituents"].labels == gold_constit_spans_flat
        span_mask = [
            1 if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        assert fields["span_mask"].labels == span_mask
        assert fields["target_index"].sequence_index == 0
