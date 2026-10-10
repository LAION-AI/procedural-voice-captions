"""Meaningful routing, calibration and timeline checks without model downloads."""
import unittest

from caption import caption_detail, _zscore
from humaneness_backend import HumanenessMedium, caption_scores, window_starts
from humaneness_captions import PARAKEET_LANGUAGES


class MediumBackendTests(unittest.TestCase):
    def test_low_blend_score_does_not_claim_burst_absence(self):
        base={'_provenance':{'model_variant':'medium'},'blend':{'mean':5,'std':1,'normalization':'mean_std'}}
        detail=caption_detail({'dims':{},'emo':{},'blend':0,'score_backend':'humaneness-medium'},base,always_on=[])
        self.assertFalse(any(q['dim']=='blend' for q in detail['quality']))

    def test_taxonomy_is_explicit_and_raw_values_are_preserved(self):
        view=caption_scores({"scores_raw":{"emo_Hope_Enthusiasm_Optimism":7.25,
            "vn_TEMP_reg":-1.5,"audiobox:CE":3.7,"vn_ESTH_reg":100}})
        self.assertEqual(view["emo"]["Hope/Optimism"],7.25)
        self.assertEqual(view["dims"]["TEMP"],-1.5)
        self.assertEqual(view["dims"]["ESTH"],100)
        self.assertEqual(view["content_enjoyment"],3.7)

    def test_medium_z_uses_mean_and_std_without_legacy_floor(self):
        stat={"mean":4,"median":9,"std":.2,"spread":10,"normalization":"mean_std"}
        self.assertAlmostEqual(_zscore(4.4,stat),2)
        self.assertEqual(_zscore(4,dict(stat,std=0)),0)

    def test_pleasantness_never_comes_from_esth(self):
        base={"_provenance":{"model_variant":"medium"},
            "ESTH":{"mean":0,"std":1,"median":0,"group":"voicenet","normalization":"mean_std"},
            "content_enjoyment":{"mean":2,"std":1,"group":"quality","normalization":"mean_std"}}
        scores={"dims":{"ESTH":100},"emo":{},"content_enjoyment":0,"score_backend":"humaneness-medium"}
        detail=caption_detail(scores,base,always_on=[])
        self.assertEqual(detail["voicenet"],[])
        self.assertEqual(detail["quality"][0]["dim"],"content_enjoyment")
        self.assertEqual(detail["quality"][0]["z"],-2)
        self.assertIn("less enjoyable",detail["quality"][0]["phrase"])

    def test_wrong_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            caption_detail({"dims":{},"score_backend":"humaneness-medium"},{})

    def test_windows_cover_the_complete_audio(self):
        for samples in [1600,480000,480001,720003,960000,1600001]:
            starts=window_starts(samples)
            self.assertEqual(starts[0],0)
            self.assertGreaterEqual(starts[-1]+480000,samples)
            self.assertTrue(all(b<=a+480000 for a,b in zip(starts,starts[1:])))

    def test_owned_events_keep_both_times_and_do_not_duplicate(self):
        import numpy as np
        model=HumanenessMedium.__new__(HumanenessMedium)
        model.names=["x"];model.mean=np.array([0.]);model.std=np.array([1.])
        def prediction(value,event):
            return {"duration_s":30,"scores_raw":{"x":value},"scores_training_z":{"x":value},
                "characters_per_second":value,"burst_probability_per_20ms_frame":[.3]*1500,
                "burst_event_proposals":[event],"orange_timbre_128d":[1.,0.],"orange_identity_250d":[0.,1.]}
        first=prediction(2,{"start_s":25,"end_s":25.5,"onset_probability":.8,"top3_classes":[]})
        last=prediction(4,{"start_s":10,"end_s":10.5,"onset_probability":.8,"top3_classes":[]})
        result=model.combine_window_predictions(720000,[0,240000],[first,last])
        self.assertEqual(len(result["burst_event_proposals"]),1)
        self.assertEqual(result["burst_event_proposals"][0]["end_s"],25.5)
        self.assertEqual(len(result["burst_probability_per_20ms_frame"]),2250)
        self.assertAlmostEqual(result["scores_raw"]["x"],3.)
        self.assertTrue(np.isfinite(result["burst_probability_per_20ms_frame"]).all())

    def test_unsupported_asr_languages_are_not_misrepresented(self):
        self.assertTrue({"de","en","fr"}<=PARAKEET_LANGUAGES)
        self.assertTrue({"ja","ko","zh"}.isdisjoint(PARAKEET_LANGUAGES))

    def test_final_fractional_frame_is_covered_for_end_anchored_windows(self):
        import math
        import numpy as np
        model=HumanenessMedium.__new__(HumanenessMedium)
        model.names=["x"];model.mean=np.array([0.]);model.std=np.array([1.])
        for samples in [480001, 640001, 720013, 800100, 960099, 1600001]:
            starts=window_starts(samples)
            prediction={"duration_s":30,"scores_raw":{"x":2},"scores_training_z":{"x":2},
                "characters_per_second":2,"burst_probability_per_20ms_frame":[.3]*1500,
                "burst_event_proposals":[],"orange_timbre_128d":[1.,0.],"orange_identity_250d":[0.,1.]}
            result=model.combine_window_predictions(samples,starts,[prediction]*len(starts))
            self.assertEqual(len(result["burst_probability_per_20ms_frame"]),math.ceil(samples/320))
            np.testing.assert_allclose(result["burst_probability_per_20ms_frame"],.3)


if __name__=="__main__":unittest.main()
