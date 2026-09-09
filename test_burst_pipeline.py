#!/usr/bin/env python3
"""Regression tests for the burst pipeline — the *old* path first, then the x2 wiring.

The repo had no tests before the x2 head was added, so the first job of this file is to
pin the behaviour that existed *before* the change: burst insertion, pause markers, the
six Variant-B templates and their seeding. If adding a second classifier had moved any of
those, the change would be a rewrite wearing an addition's clothes.

Everything here runs without torch, without numpy and without a model: `burst_captions`
imports its heavy dependencies inside the functions that need them, so the composition
layer can be exercised on a bare interpreter. The few tests that genuinely need numpy
(`extract_events`) skip themselves.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import augment                                                            # noqa: E402
import burst_captions as B                                                # noqa: E402
import burst_x2                                                           # noqa: E402

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

WORDS = [{"w": "Hi", "start": 0.0, "end": 0.30},
         {"w": "there", "start": 1.00, "end": 1.40},
         {"w": "friend", "start": 1.45, "end": 1.90}]


def _captioner(x2=None, table=None, label_source="v2"):
    """A BurstCaptioner shell with no models — enough for the composition layer."""
    bc = object.__new__(B.BurstCaptioner)
    bc.verbose = False
    bc.x2 = x2
    bc.x2_table = table
    bc.x2_head = "large-v2"
    bc.label_source = label_source
    bc.label_group = {}
    return bc


# --------------------------------------------------------------------------- #
class TestUnchangedBehaviour(unittest.TestCase):
    """What the repo did before the x2 head, and must still do with it switched off."""

    def test_variant_a_places_a_burst_between_the_nearest_words(self):
        out = B.BurstCaptioner.variant_a(WORDS, [(0.40, 0.60, "Chuckle", 0.8)])
        self.assertEqual(out, "Hi (Chuckle) there friend")

    def test_variant_a_with_no_timestamps_emits_bursts_in_time_order(self):
        out = B.BurstCaptioner.variant_a([], [(2.0, 2.2, "Yawn", 0.5), (0.1, 0.2, "Scream", 0.9)])
        self.assertEqual(out, "(Scream) (Yawn)")

    def test_pause_markers_appear_on_a_silent_gap_the_burst_did_not_fill(self):
        out = B.BurstCaptioner.variant_a(WORDS, [])
        self.assertIn("[pause 0.7s]", out)
        # ... and are suppressed where a burst already accounts for the silence
        out2 = B.BurstCaptioner.variant_a(WORDS, [(0.40, 0.60, "Chuckle", 0.8)])
        self.assertNotIn("[pause", out2)

    def test_the_six_variant_b_templates_are_unchanged_and_still_seeded_the_same(self):
        # Pinned against the implementation as it stood before the x2 head existed.
        self.assertEqual(burst_x2.PHRASES_NAMED,
                         ["punctuated by a ({b})", "with an audible ({b})", "broken by a ({b})",
                          "carrying a ({b})", "interrupted by a ({b})", "marked by a ({b})"])
        for (lab, seed), want in {
                ("Chuckle", 0): "interrupted by a (Chuckle)",
                ("Chuckle", 7): "with an audible (Chuckle)",
                ("Contented Sigh", 3): "marked by a (Contented Sigh)",
                ("Ahem", 12345): "interrupted by a (Ahem)",
                ("Scream", 99): "carrying a (Scream)"}.items():
            self.assertEqual(B.BurstCaptioner.variant_b_phrase(lab, seed), want, (lab, seed))

    def test_variant_b_phrase_still_takes_two_arguments(self):
        # build_burst_demo_v2.py calls it positionally with (label, seed).
        self.assertTrue(B.BurstCaptioner.variant_b_phrase("Yawn", 5).endswith("(Yawn)"))

    def test_the_label_source_defaults_to_v2(self):
        self.assertEqual(B.LABEL_SOURCE, "v2")
        self.assertFalse(B.USE_X2)

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_extract_events_post_processing_is_unchanged(self):
        import numpy as np
        probs = np.zeros(200, dtype="float32")
        probs[10:25] = 0.9          # 0.30 s run
        probs[28:40] = 0.8          # 0.24 s run, 0.06 s after the first -> merged (gap 0.10)
        probs[120:122] = 0.9        # 0.04 s -> below the 0.10 s floor, dropped
        ev = B.extract_events(probs)
        self.assertEqual(len(ev), 1)
        s, e, peak = ev[0]
        self.assertAlmostEqual(s, 0.20, places=3)
        self.assertAlmostEqual(e, 0.80, places=3)
        self.assertAlmostEqual(peak, 0.9, places=3)


# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestWordEndTrimming(unittest.TestCase):
    """A TDT token's duration runs to the NEXT token, so a word span swallows the pause after
    it. `trim_word_ends` gives the span back its real end. Measured on 96 real clips: without
    it, 233 of 313 audible pauses are printed and those are understated by a mean 0.134 s."""

    def _clip(self, segments, dur=3.0, sr=B.SR):
        """A waveform that is loud inside `segments` and digitally silent elsewhere."""
        import numpy as np
        rng = np.random.default_rng(0)
        x = np.zeros(int(dur * sr), dtype="float32")
        for a, b in segments:
            i0, i1 = int(a * sr), int(b * sr)
            x[i0:i1] = rng.standard_normal(i1 - i0).astype("float32") * 0.3
        return x

    def test_a_span_that_overruns_into_silence_is_trimmed_back(self):
        wav = self._clip([(0.0, 0.40), (1.50, 1.90)])
        words = [{"w": "one", "start": 0.0, "end": 1.50},     # TDT: runs to the next token
                 {"w": "two", "start": 1.50, "end": 1.90}]
        got = B.trim_word_ends(wav, words)
        self.assertAlmostEqual(got[0]["end"], 0.40, delta=0.03)
        self.assertEqual(got[0]["end_span"], 1.50)            # the raw span is kept
        self.assertAlmostEqual(got[1]["end"], 1.90, delta=0.03)

    def test_the_pause_only_becomes_visible_after_trimming(self):
        wav = self._clip([(0.0, 0.40), (1.50, 1.90)])
        words = [{"w": "one", "start": 0.0, "end": 1.50},
                 {"w": "two", "start": 1.50, "end": 1.90}]
        self.assertNotIn("[pause", B.BurstCaptioner.variant_a(words, []))     # gap == 0.000
        self.assertIn("[pause 1.1s]",
                      B.BurstCaptioner.variant_a(B.trim_word_ends(wav, words), []))

    def test_a_genuinely_long_word_is_not_shortened(self):
        # This is exactly where a blind 0.30 s cap invents a pause: measured 154 false
        # markers out of 431 on real audio. The energy rule leaves the word alone.
        wav = self._clip([(0.0, 0.90), (1.00, 1.40)])
        words = [{"w": "Aufnahmebereitschaft", "start": 0.0, "end": 0.90},
                 {"w": "der", "start": 1.00, "end": 1.40}]
        got = B.trim_word_ends(wav, words)
        self.assertAlmostEqual(got[0]["end"], 0.90, delta=0.03)
        self.assertNotIn("[pause", B.BurstCaptioner.variant_a(got, []))

    def test_an_end_never_moves_later_and_never_before_its_own_start(self):
        wav = self._clip([(0.0, 2.9)])
        words = [{"w": "a", "start": 0.5, "end": 0.9}, {"w": "b", "start": 0.9, "end": 1.3}]
        for w0, w1 in zip(words, B.trim_word_ends(wav, words)):
            self.assertLessEqual(w1["end"], w0["end"] + 1e-9)
            self.assertGreaterEqual(w1["end"], w1["start"])

    def test_a_word_with_no_voiced_frame_is_left_alone(self):
        wav = self._clip([(2.0, 2.5)])
        words = [{"w": "ghost", "start": 0.1, "end": 0.5}]
        self.assertEqual(B.trim_word_ends(wav, words)[0]["end"], 0.5)

    def test_raw_mode_restores_the_previous_behaviour(self):
        wav = self._clip([(0.0, 0.40), (1.50, 1.90)])
        words = [{"w": "one", "start": 0.0, "end": 1.50}]
        got = B.trim_word_ends(wav, words, mode="raw")
        self.assertEqual(got[0]["end"], 1.50)
        self.assertEqual(got[0]["end_span"], 1.50)

    def test_end_span_is_always_present_so_callers_can_rely_on_it(self):
        wav = self._clip([(0.0, 0.40)])
        for mode in ("energy", "raw"):
            got = B.trim_word_ends(wav, [{"w": "x", "start": 0.0, "end": 1.0}], mode=mode)
            self.assertIn("end_span", got[0])

    def test_the_input_list_is_not_mutated(self):
        wav = self._clip([(0.0, 0.40), (1.50, 1.90)])
        words = [{"w": "one", "start": 0.0, "end": 1.50}]
        B.trim_word_ends(wav, words)
        self.assertEqual(words[0]["end"], 1.50)
        self.assertNotIn("end_span", words[0])

    def test_a_silent_clip_changes_nothing(self):
        import numpy as np
        wav = np.zeros(int(3.0 * B.SR), dtype="float32")
        words = [{"w": "one", "start": 0.0, "end": 1.5}]
        self.assertEqual(B.trim_word_ends(wav, words)[0]["end"], 1.5)

    def test_frame_rms_tracks_the_envelope(self):
        import numpy as np
        wav = self._clip([(1.0, 2.0)])
        r = B.frame_rms(wav)
        self.assertGreater(r[int(1.5 / B.RMS_HOP)], 10 * (r[int(0.5 / B.RMS_HOP)] + 1e-9))
        self.assertAlmostEqual(len(r) * B.RMS_HOP, 3.0, delta=0.05)


class TestBurstAssignment(unittest.TestCase):
    SENTS = [{"text": "Hi there friend.", "start": 0.0, "end": 2.0, "caption": "warm"},
             {"text": "And again.", "start": 2.5, "end": 4.0, "caption": "flat"}]

    def test_every_kept_burst_is_written_exactly_once(self):
        bc = _captioner()
        kept = [{"start": 0.5, "end": 0.7, "label": "Chuckle", "prob": 0.8},
                {"start": 3.0, "end": 3.2, "label": "Yawn", "prob": 0.6}]
        lines = bc._assign_bursts([dict(s) for s in self.SENTS], WORDS, kept)
        self.assertEqual([l["bursts"] for l in lines], [["Chuckle"], ["Yawn"]])
        joined = " ".join(l["text"] for l in lines)
        self.assertEqual(joined.count("(Chuckle)"), 1)
        self.assertEqual(joined.count("(Yawn)"), 1)

    def test_the_written_form_is_what_lands_in_the_script(self):
        bc = _captioner()
        kept = [{"start": 0.5, "end": 0.7, "label": "Deep Breath", "prob": 0.8,
                 "written": "Breath", "vocab": "x2-17-family"}]
        lines = bc._assign_bursts([dict(s) for s in self.SENTS], WORDS, kept)
        self.assertIn("(Breath)", lines[0]["text"])
        self.assertNotIn("Deep Breath", lines[0]["text"])
        self.assertEqual(lines[0]["burst_vocab"], ["x2-17-family"])

    def test_records_without_a_written_field_still_work(self):
        # Anything produced before this change, or by an external caller.
        bc = _captioner()
        kept = [{"start": 0.5, "end": 0.7, "label": "Chuckle", "prob": 0.8}]
        lines = bc._assign_bursts([dict(s) for s in self.SENTS], WORDS, kept)
        self.assertIn("(Chuckle)", lines[0]["text"])

    def test_burst_vocab_follows_the_same_order_as_the_bursts(self):
        bc = _captioner()
        kept = [{"start": 1.6, "end": 1.7, "label": "Yawn", "prob": 0.5,
                 "written": "Yawn?", "vocab": "x2-17"},
                {"start": 0.4, "end": 0.5, "label": "Chuckle", "prob": 0.9,
                 "written": "Chuckle", "vocab": "v2-83"}]
        lines = bc._assign_bursts([dict(s) for s in self.SENTS], WORDS, kept)
        self.assertEqual(lines[0]["bursts"], ["Chuckle", "Yawn?"])
        self.assertEqual(lines[0]["burst_vocab"], ["v2-83", "x2-17"])


# --------------------------------------------------------------------------- #
class TestMergeOnTheCaptioner(unittest.TestCase):
    def test_with_no_x2_head_every_span_is_v2_vocabulary(self):
        bc = _captioner()
        m = bc.merge_burst_label("Contented Sigh", None)
        self.assertEqual(m, {"label": "Contented Sigh", "written": "Contented Sigh",
                             "vocab": "v2-83", "tier": "named", "source": "v2",
                             "policy": "v2", "out_of_x2_vocab": None})

    def test_a_rejected_span_writes_nothing(self):
        for bc in (_captioner(),
                   _captioner(x2=object(), table=burst_x2.RecallTable.load("large-v2"),
                              label_source="union")):
            self.assertIsNone(bc.merge_burst_label(None, {"label": "Chuckle", "prob": 0.9})
                              ["written"])

    def test_union_on_the_captioner_matches_the_pure_function(self):
        t = burst_x2.RecallTable.load("large-v2")
        bc = _captioner(x2=object(), table=t, label_source="union")
        x2 = {"label": "Deep Breath", "prob": 0.4}
        self.assertEqual(bc.merge_burst_label("Deep Breath", x2),
                         burst_x2.merge_label("Deep Breath", x2, t, "union"))

    def test_x2_meta_says_so_when_the_head_is_off(self):
        meta = _captioner().x2_meta()
        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["label_source"], "v2")
        self.assertEqual(meta["n_classes"], 83)

    def test_x2_fields_are_empty_without_a_result_and_complete_with_one(self):
        self.assertEqual(B.BurstCaptioner._x2_fields(None), {})
        f = B.BurstCaptioner._x2_fields(
            {"label": "Deep Breath", "prob": 0.42, "labels": [("Deep Breath", 0.42)],
             "family": "breath", "no_burst_prob": 0.03, "tier": "family",
             "written": "Breath", "recall": 0.1864, "family_recall": 0.7119})
        self.assertEqual(f["x2_label"], "Deep Breath")
        self.assertEqual(f["x2_written"], "Breath")
        self.assertEqual(f["x2_tier"], "family")
        self.assertEqual(f["x2_top"], [["Deep Breath", 0.42]])


# --------------------------------------------------------------------------- #
class TestAugmentRoundTrip(unittest.TestCase):
    def _result(self, bursts):
        return {"id": "clip", "ei_gender": 1.0,
                "scores": {"dims": {"WARM": 5.0}, "emo": {"Amusement": 4.0},
                           "genu": 3.0, "blend": 5.0},
                "sentences": [{"text": "Hi there friend.", "start": 0.0, "end": 2.0,
                               "scores": {"dims": {"WARM": 5.0}, "emo": {}, "genu": 3.0,
                                          "blend": 5.0}}],
                "variant_a_bursts": bursts,
                "x2": {"enabled": True, "head": "large-v2", "label_source": "union"}}

    def test_score_record_stores_the_written_form_and_its_provenance(self):
        rec = augment.score_record(self._result(
            [{"start": 0.5, "end": 0.7, "label": "Deep Breath", "written": "Breath",
              "vocab": "x2-17-family", "tier": "family", "kept": True},
             {"start": 0.9, "end": 1.0, "label": "Yawn", "kept": False}]))
        self.assertEqual(rec["sentences"][0]["bursts"],
                         [{"written": "Breath", "label": "Deep Breath",
                           "vocab": "x2-17-family", "tier": "family"}])
        self.assertEqual(rec["x2"]["head"], "large-v2")

    def test_augment_script_writes_the_written_form(self):
        rec = augment.score_record(self._result(
            [{"start": 0.5, "end": 0.7, "label": "Deep Breath", "written": "Breath",
              "vocab": "x2-17-family", "tier": "family", "kept": True}]))
        txt = augment.augment_script(rec, seed=1, template="tags")
        self.assertIn("(Breath)", txt)
        self.assertNotIn("(Deep Breath)", txt)

    def test_records_written_before_the_x2_head_still_caption(self):
        # Bursts stored as bare strings — the format augment.py shipped with.
        rec = {"ei_gender": 1.0, "global_scores": {"dims": {"WARM": 5.0}, "emo": {},
                                                   "genu": 3.0, "blend": 5.0},
               "sentences": [{"text": "Hi.", "start": 0.0, "end": 1.0,
                              "scores": {"dims": {"WARM": 5.0}, "emo": {}, "genu": 3.0,
                                         "blend": 5.0},
                              "bursts": ["Contented Sigh"]}]}
        txt = augment.augment_script(rec, seed=2, template="tags")
        self.assertIn("(Contented Sigh)", txt)

    def test_a_result_with_no_x2_block_produces_a_record_with_no_x2_key(self):
        r = self._result([])
        r.pop("x2")
        self.assertNotIn("x2", augment.score_record(r))


if __name__ == "__main__":
    unittest.main()
