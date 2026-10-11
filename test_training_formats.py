import unittest
from training_formats import moss_training_formats
from prompt_format import numeric_budget


class TrainingFormatTest(unittest.TestCase):
    def test_six_variants_keep_the_original_caption_and_actual_burst_endpoints(self):
        proc={'transcript':'Hello "there".', 'words':[{'w':'Hello','start':0.,'end':.4},
               {'w':'"there".','start':.7,'end':1.} ]}
        moss={'voice_caption':{'prediction':'A warm voice with a chuckle.'},'audio_duration_s':1.2,
              'locator_settings':{'threshold':.6},'burst_captions':[{'start_s':.2,'end_s':.3,
              'top3_classes':[{'label':'Chuckle','softmax_probability':.8}],
              'caption':{'prediction':'A brief breathy chuckle.'}}]}
        result=moss_training_formats(proc,moss)
        self.assertEqual(len(result['variants']),6)
        for name,v in result['variants'].items():
            self.assertTrue(v['prompt'].startswith('CAPTION: '))
            self.assertIn('\nTRANSCRIPT: "',v['prompt'])
            if name.startswith('whole_audio_only'):self.assertEqual(v['caption'],moss['voice_caption']['prediction'])
            else:self.assertIn('0.20–0.30 s (0.10 s)',v['caption'])
            if name.endswith('__plain'):self.assertEqual(v['transcript'],proc['transcript'])
            else:self.assertIn('(Chuckle, 0.10 seconds)',v['transcript'])
        self.assertAlmostEqual(numeric_budget(result['variants']['whole_audio_only__timed']['transcript']),1.2)


if __name__=='__main__':unittest.main()
