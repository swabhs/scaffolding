# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers.ontonotes.syntactic_constituent_reader import SyntacticConstitReader
from allennlp.common.testing import AllenNlpTestCase


class TestSyntacticConstitReader(AllenNlpTestCase):

    def test_read_from_file(self):
        max_span_width = 10
        reader = SyntacticConstitReader(max_span_width)
        dataset = reader.read('tests/fixtures/conll_2012/')
        instances = dataset.instances

        fields = instances[0].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ["Mali", "government", "officials", "say", "the", "woman", "'s",
                          "confession", "was", "forced", "."]
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
        gold_constit_spans_flat = [gs for rows in gold_constit_spans for gs in rows]
        assert fields["tags"].labels == gold_constit_spans_flat
        span_mask = [
            1 if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        assert fields["span_mask"].labels == span_mask

        fields = instances[1].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month', 'after',
                          'four', 'months', 'of', 'hearings', '.']
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
        gold_constit_spans_flat = [gs for rows in gold_constit_spans for gs in rows]
        assert fields["tags"].labels == gold_constit_spans_flat
        span_mask = [
            1 if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        assert fields["span_mask"].labels == span_mask

        fields = instances[2].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        assert tokens == ['Denise', 'Dillon', 'Headline', 'News', '.']
        gold_constit_spans = [['*' for d in range(max_span_width)]
                      for j in range(len(tokens))]
        gold_constit_spans[1][1] = 'NP'
        gold_constit_spans[3][1] = 'NP'
        gold_constit_spans[4][4] = 'FRAG|TOP'
        gold_constit_spans_flat = [gs for rows in gold_constit_spans for gs in rows]
        assert fields["tags"].labels == gold_constit_spans_flat
        span_mask = [
            1 if j-d >= 0 else 0 for j in range(len(tokens)) for d in range(max_span_width)]
        assert fields["span_mask"].labels == span_mask
