#!/usr/bin/env python3
"""Tests for the x2 head's *decision* layer — the half that decides what a caption may say.

Deliberately model-free and stdlib-only: `RecallTable`, the confidence tiers and
`merge_label` are the parts that change captions, and they must be checkable without a
GPU, without a download and without 101 s of encoder load. Run them with

    python -m unittest discover -v

The two most important tests here are not about arithmetic:

* `test_x2_classes_are_a_strict_subset_of_the_taxonomy` pins the finding that motivated
  the whole design — the 16 x2 classes add nothing and remove 66.
* `test_union_keeps_labels_the_x2_head_cannot_say` pins the consequence — the four most
  common burst labels in the corpus survive the merge.

If either ever fails, the integration has turned into a replacement and the caption
coverage is about to collapse.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import burst_x2                                                          # noqa: E402
from burst_x2 import RecallTable, merge_label, phrases_for               # noqa: E402

# Labels that dominate the existing corpus and that the 16-class head cannot emit.
# Measured on 61,579 rows / 12,894 burst events of laion-tts-annotated-v1.
CORPUS_TOP_OLD = ["Low Mumble", "Ahem", "Contented Sigh", "Surprised Gasp"]


class TestRecallTable(unittest.TestCase):
    def setUp(self):
        self.t = RecallTable.load("large-v2")

    def test_both_bundled_tables_load(self):
        for head in ("large-v2", "commercial"):
            t = RecallTable.load(head)
            self.assertEqual(len(t.classes), 17)
            self.assertEqual(len(t.bursts), 16)
            self.assertIn("no_burst", t.classes)
            self.assertAlmostEqual(t.chance, 1 / 17, places=3)

    def test_encoders_are_the_documented_ones(self):
        self.assertEqual(RecallTable.load("large-v2").encoder, "voiceclap-large-v2")
        self.assertEqual(RecallTable.load("commercial").encoder, "voiceclap-commercial")

    def test_unknown_head_is_refused(self):
        with self.assertRaises(ValueError):
            RecallTable.load("small-v2")      # CC BY-NC — deliberately not bundled

    def test_families_cover_every_burst_class(self):
        fams = {self.t.family(c) for c in self.t.bursts}
        self.assertEqual(fams, {"breath", "sigh", "groan", "laugh", "hum", "scream", "yawn"})
        for c in self.t.bursts:
            self.assertIsNotNone(self.t.family(c))

    def test_recall_of_an_unknown_label_is_none_not_zero(self):
        # None means "outside this vocabulary"; 0.0 would mean "measured and hopeless".
        for lab in CORPUS_TOP_OLD:
            self.assertIsNone(self.t.strict(lab), lab)
            self.assertFalse(self.t.in_vocab(lab), lab)

    def test_min_source_is_the_worse_of_the_two_measurements(self):
        real = RecallTable.load("large-v2", source="real")
        dbox = RecallTable.load("large-v2", source="dramabox")
        worst = RecallTable.load("large-v2", source="min")
        for c in worst.bursts:
            self.assertAlmostEqual(worst.strict(c), min(real.strict(c), dbox.strict(c)), places=6)


class TestTiers(unittest.TestCase):
    def setUp(self):
        self.t = RecallTable.load("large-v2")

    def test_the_bundled_large_v2_table_tiers_as_measured(self):
        # Pins both the table and the default thresholds (0.50 / 0.25 / 0.50).
        s = self.t.summary()
        self.assertEqual(sorted(s["named"]),
                         ["Affirmative Grunt", "Chuckle", "Frustrated Groan", "Panting",
                          "Scream", "Sharp Inhale"])
        self.assertEqual(sorted(s["hedged"]),
                         ["Breathy Giggle", "Exhausted Groan", "Humming", "Soft Hum", "Yawn"])
        self.assertEqual(sorted(s["family"]), ["Deep Breath", "Heavy Breathing"])
        self.assertEqual(sorted(s["generic"]),
                         ["Exasperated Sigh", "Relief Sigh", "Wistful Sigh"])
        self.assertEqual(sum(len(v) for v in s.values()), 16)

    def test_the_commercial_table_tiers_differently(self):
        s = RecallTable.load("commercial").summary()
        self.assertEqual({k: len(v) for k, v in s.items()},
                         {"named": 6, "hedged": 4, "family": 2, "generic": 4})

    def test_written_form_per_tier(self):
        self.assertEqual(self.t.write("Sharp Inhale"), "Sharp Inhale")      # named, 0.897
        self.assertEqual(self.t.write("Yawn"), "Yawn?")                     # hedged, 0.460
        self.assertEqual(self.t.write("Deep Breath"), "Breath")             # family, 0.186/0.712
        self.assertEqual(self.t.write("Relief Sigh"), "Vocal Burst")        # generic, 0.062

    def test_thresholds_are_inclusive_at_the_boundary(self):
        t = RecallTable.load("large-v2", strict_min=0.4595, hedge_min=0.1864, family_min=0.7119)
        self.assertEqual(t.tier("Yawn"), "named")            # strict == strict_min
        self.assertEqual(t.tier("Deep Breath"), "hedged")    # strict == hedge_min
        t2 = RecallTable.load("large-v2", strict_min=0.9, hedge_min=0.9, family_min=0.7119)
        self.assertEqual(t2.tier("Deep Breath"), "family")   # family_recall == family_min

    def test_tightening_the_thresholds_never_upgrades_a_class(self):
        order = {"named": 0, "hedged": 1, "family": 2, "generic": 3}
        loose = RecallTable.load("large-v2", strict_min=0.10, hedge_min=0.05, family_min=0.10)
        tight = RecallTable.load("large-v2", strict_min=0.90, hedge_min=0.80, family_min=0.90)
        for c in loose.bursts:
            self.assertLessEqual(order[loose.tier(c)], order[tight.tier(c)], c)

    def test_a_singleton_family_can_never_reach_the_family_tier(self):
        # `scream` and `yawn` have one member each, so family recall == strict recall and a
        # class that failed the hedge threshold cannot pass the family one. Not a special
        # case in the code — a property of the tiering, asserted so it stays true.
        t = RecallTable.load("large-v2", strict_min=0.99, hedge_min=0.99, family_min=0.10)
        for c in ("Scream", "Yawn"):
            self.assertEqual(t.tier(c), "family" if t.family_recall(c) >= 0.10 else "generic")
            self.assertEqual(t.family_word(c), c.split()[-1].capitalize())

    def test_describe_carries_the_provenance_a_reviewer_needs(self):
        d = self.t.describe("Deep Breath")
        self.assertEqual(d["tier"], "family")
        self.assertEqual(d["written"], "Breath")
        self.assertEqual(d["recall_source"], "real")
        self.assertEqual(d["family"], "breath")
        self.assertIsInstance(d["n_heldout"], int)
        self.assertIsInstance(d["recall_reliable"], bool)

    def test_hedge_mark_and_generic_label_are_configurable(self):
        t = RecallTable.load("large-v2", hedge_mark=" (uncertain)", generic_label="Non-verbal")
        self.assertEqual(t.write("Yawn"), "Yawn (uncertain)")
        self.assertEqual(t.write("Relief Sigh"), "Non-verbal")


class TestVocabularyRelation(unittest.TestCase):
    """The finding the whole integration is built around."""

    def setUp(self):
        with open(os.path.join(HERE, "vocalburst_taxonomy.json"), encoding="utf-8") as f:
            tax = json.load(f)["categories"]
        self.old = [n for g, d in tax.items() for n in d.get("items", {})]
        self.group = {n: g for g, d in tax.items() for n in d.get("items", {})}
        self.t = RecallTable.load("large-v2")

    def test_the_old_taxonomy_is_the_82_classes_the_v2_head_names(self):
        self.assertEqual(len(self.old), 82)
        self.assertEqual(len(set(self.old)), 82)

    def test_x2_classes_are_a_strict_subset_of_the_taxonomy(self):
        # Every x2 class already exists in the 82-class taxonomy: the new head adds no
        # vocabulary at all, it removes 66 classes.
        for c in self.t.bursts:
            self.assertIn(c, self.old, f"{c} is not in vocalburst_taxonomy.json")
        self.assertEqual(len(self.old) - len(self.t.bursts), 66)

    def test_seven_taxonomy_groups_become_unsayable(self):
        covered = {self.group[c] for c in self.t.bursts}
        lost = sorted(set(self.group.values()) - covered)
        self.assertEqual(lost, ["crying_and_distress", "eating_and_drinking",
                                "hand_and_body_sounds", "mouth_and_lip_sounds",
                                "throat_and_vocal_sounds", "tongue_clicks", "whistling"])

    def test_the_four_commonest_corpus_labels_are_all_outside_x2(self):
        for lab in CORPUS_TOP_OLD:
            self.assertIn(lab, self.old, lab)              # the v2 head can say it
            self.assertNotIn(lab, self.t.bursts, lab)      # the x2 head cannot


class TestMergePolicies(unittest.TestCase):
    def setUp(self):
        self.t = RecallTable.load("large-v2")
        self.x2_named = {"label": "Chuckle", "prob": 0.8}
        self.x2_generic = {"label": "Relief Sigh", "prob": 0.6}
        self.x2_family = {"label": "Deep Breath", "prob": 0.5}

    def test_v2_policy_is_the_shipped_behaviour_untouched(self):
        for x2 in (None, self.x2_named, self.x2_generic):
            m = merge_label("Contented Sigh", x2, self.t, policy="v2")
            self.assertEqual(m["written"], "Contented Sigh")
            self.assertEqual(m["vocab"], "v2-83")
            self.assertEqual(m["source"], "v2")

    def test_x2_policy_renames_even_out_of_vocabulary_spans(self):
        m = merge_label("Ahem", self.x2_named, self.t, policy="x2")
        self.assertEqual(m["written"], "Chuckle")
        self.assertEqual(m["vocab"], "x2-17")
        self.assertTrue(m["out_of_x2_vocab"])
        self.assertFalse(m["agrees_with_v2"])

    def test_union_keeps_labels_the_x2_head_cannot_say(self):
        # The whole point: coverage of the 82-class vocabulary is preserved.
        for lab in CORPUS_TOP_OLD:
            m = merge_label(lab, self.x2_named, self.t, policy="union")
            self.assertEqual(m["written"], lab, lab)
            self.assertEqual(m["vocab"], "v2-83", lab)
            self.assertTrue(m["out_of_x2_vocab"], lab)

    def test_union_uses_x2_where_both_heads_have_jurisdiction(self):
        m = merge_label("Chuckle", {"label": "Breathy Giggle", "prob": 0.7},
                        self.t, policy="union")
        self.assertEqual(m["written"], "Breathy Giggle?")     # hedged, recall 0.469
        self.assertEqual(m["vocab"], "x2-17")
        self.assertEqual(m["tier"], "hedged")
        self.assertFalse(m["out_of_x2_vocab"])

    def test_union_backs_off_to_the_family_and_to_generic(self):
        m = merge_label("Deep Breath", self.x2_family, self.t, policy="union")
        self.assertEqual(m["written"], "Breath")
        self.assertEqual(m["vocab"], "x2-17-family")
        m = merge_label("Relief Sigh", self.x2_generic, self.t, policy="union")
        self.assertEqual(m["written"], "Vocal Burst")
        self.assertEqual(m["vocab"], "generic")

    def test_agreement_between_the_heads_is_recorded(self):
        m = merge_label("Chuckle", {"label": "Chuckle", "prob": 0.9}, self.t, policy="union")
        self.assertTrue(m["agrees_with_v2"])

    def test_an_x2_no_burst_verdict_never_writes_the_caption(self):
        m = merge_label("Chuckle", {"label": "no_burst", "prob": 0.9}, self.t, policy="x2")
        self.assertEqual(m["written"], "Chuckle")
        self.assertEqual(m["vocab"], "v2-83")

    def test_a_rejected_span_stays_rejected_under_every_policy(self):
        for pol in ("v2", "x2", "union"):
            m = merge_label(None, self.x2_named, self.t, policy=pol)
            self.assertIsNone(m["written"], pol)

    def test_an_unknown_policy_is_refused_loudly(self):
        with self.assertRaises(ValueError):
            merge_label("Chuckle", self.x2_named, self.t, policy="best")

    def test_every_written_form_is_bracket_safe(self):
        # The demo pages highlight bursts with re.sub(r"\\(([^)]+)\\)"), so a written form
        # containing a bracket would silently break the rendering.
        for c in self.t.bursts:
            w = self.t.write(c)
            self.assertNotIn("(", w)
            self.assertNotIn(")", w)
            self.assertTrue(w.strip())


class TestPhrases(unittest.TestCase):
    def test_only_the_hedged_tier_changes_the_wording(self):
        self.assertIs(phrases_for("named"), burst_x2.PHRASES_NAMED)
        self.assertIs(phrases_for("family"), burst_x2.PHRASES_NAMED)
        self.assertIs(phrases_for("generic"), burst_x2.PHRASES_NAMED)
        self.assertIs(phrases_for("hedged"), burst_x2.PHRASES_HEDGED)

    def test_the_two_template_sets_stay_index_aligned(self):
        # Same length means the same seed picks the same slot, so turning hedging on
        # rephrases a cue without reshuffling which cue it was.
        self.assertEqual(len(burst_x2.PHRASES_NAMED), len(burst_x2.PHRASES_HEDGED))
        for t in burst_x2.PHRASES_NAMED + burst_x2.PHRASES_HEDGED:
            self.assertIn("({b})", t)


if __name__ == "__main__":
    unittest.main()
