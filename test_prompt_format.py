"""Regression cases for timing/data loss observed in the caption collection."""
import unittest
from prompt_format import render_timed, verify_timed, numeric_budget, TimingUnavailable


class TimingTests(unittest.TestCase):
    def test_shared_sentence_boundary_does_not_duplicate_words(self):
        words = [{'w':'Yes.', 'start':0., 'end':.5}, {'w':'Really?', 'start':.5, 'end':1.}]
        sentences = [{'text':'Yes.', 'start':0., 'end':.5}, {'text':'Really?', 'start':.5, 'end':1.}]
        lines, meta = render_timed(sentences, words, [], 1.)
        self.assertEqual(meta['n_words'], 2)
        self.assertEqual(verify_timed('\n'.join(lines), 'Yes. Really?', 1.), [])

    def test_overlap_retains_full_duration_without_double_count(self):
        words = [{'w':'Really?', 'start':0., 'end':1.}]
        lines, meta = render_timed([], words, [{'start':.2,'end':.5,'label':'Ahem'}], 1.)
        self.assertIn('(Ahem, 0.30 seconds)', '\n'.join(lines))
        self.assertEqual(meta['n_inline_overlapping_events'], 1)
        self.assertAlmostEqual(numeric_budget('\n'.join(lines)), 1.)

    def test_burst_in_gap_is_not_called_a_pause(self):
        words = [{'w':'Hi.', 'start':0., 'end':.4}, {'w':'Again.', 'start':1., 'end':1.5}]
        lines, meta = render_timed([], words, [{'start':.5,'end':.8,'label':'Chuckle'}], 2.)
        self.assertIn('(Chuckle, 0.30 seconds)', lines)
        self.assertIn('[0.20 seconds pause]', lines)
        self.assertEqual(verify_timed('\n'.join(lines),'Hi. Again.',2.,expected_bursts=1), [])

    def test_overlapping_bursts_without_words(self):
        bursts = [{'start':.2,'end':.6,'label':'Chuckle'},{'start':.4,'end':.9,'label':'Breath'}]
        lines, meta = render_timed([], [], bursts, 1.)
        self.assertIn('(Chuckle, 0.40 seconds)', '\n'.join(lines))
        self.assertIn('(Breath, 0.50 seconds)', '\n'.join(lines))
        self.assertAlmostEqual(numeric_budget('\n'.join(lines)), 1.)

    def test_fractional_rounding_cannot_accumulate_drift(self):
        words = [{'w':'One.', 'start':.033,'end':.477}, {'w':'Two.', 'start':.733,'end':1.177}]
        lines, _ = render_timed([],words,[],1.433)
        self.assertAlmostEqual(numeric_budget('\n'.join(lines)),1.43)

    def test_invalid_word_endpoint_is_not_silently_rendered_untimed(self):
        with self.assertRaises(TimingUnavailable):
            render_timed([], [{'w':'Hello', 'start':.1}], [], 1.)

    def test_round_duration_is_supported_without_confusing_bursts(self):
        lines,_=render_timed([], [{'w':'Hello.', 'start':0.,'end':1.}],[],1.,duration_style='round')
        self.assertIn('(1.00 seconds duration)', '\n'.join(lines))
        self.assertEqual(verify_timed('\n'.join(lines),'Hello.',1.,expected_bursts=0), [])


if __name__ == '__main__':
    unittest.main()
