#!/usr/bin/env python3
"""Tests for the timed-script builder + verifier in burst_captions.py (no GPU)."""
import sys
sys.path.insert(0, '/e/scratch/reformo/schuhmann1_moss/gh/procedural-voice-captions-github')
import burst_captions as bc


def case_1():
    # two sentences, a burst between words, leading/trailing sub-threshold silence
    words = [{'w': 'Hello', 'start': 0.05, 'end': 0.5},
             {'w': 'there.', 'start': 0.6, 'end': 1.0},
             {'w': 'How', 'start': 1.15, 'end': 1.5},
             {'w': 'are', 'start': 1.6, 'end': 1.8},
             {'w': 'you?', 'start': 1.9, 'end': 2.3}]
    sents = [{'text': 'Hello there.', 'start': 0.05, 'end': 1.0, 'caption': 'very warm, close, soft'},
             {'text': 'How are you?', 'start': 1.15, 'end': 2.3, 'caption': 'brisk, curious'}]
    bursts = [{'start': 1.02, 'end': 1.12, 'label': 'Chuckle'}]  # between sentences
    lines = bc.build_timed_script(sents, words, bursts, 2.4)
    script = '\n'.join(lines)
    print(script)
    bad = bc.verify_timed_script(script, 'Hello there. How are you?', 2.4)
    assert not bad, bad
    assert '[1.00 seconds duration] Hello there.' in lines[0]  # leading 0.05 folded in
    assert '(Chuckle, 0.10 seconds)' in lines[1]
    print('case1 PASS')


def case_2():
    # big pause must print; burst overlapping speech collapses to bare label
    words = [{'w': 'Yes.', 'start': 0.0, 'end': 0.4},
             {'w': 'Really', 'start': 1.0, 'end': 1.4},
             {'w': 'now?', 'start': 1.5, 'end': 1.9}]
    sents = [{'text': 'Yes.', 'start': 0.0, 'end': 0.4, 'caption': 'flat'},
             {'text': 'Really now?', 'start': 1.0, 'end': 1.9, 'caption': 'sharp'}]
    bursts = [{'start': 1.05, 'end': 1.35, 'label': 'Ahem'}]  # overlaps word 'Really' fully
    lines = bc.build_timed_script(sents, words, bursts, 2.0)
    script = '\n'.join(lines)
    print(script)
    bad = bc.verify_timed_script(script, 'Yes. Really now?', 2.0)
    assert not bad, bad
    assert any('[0.60 seconds pause]' in l for l in lines), lines
    assert '(Ahem)' in '\n'.join(lines) and '(Ahem, 0' not in '\n'.join(lines)
    print('case2 PASS')


def case_3():
    # no timestamps -> ValueError, never silent guessing
    try:
        bc.build_timed_script([{'text': 'x'}], [], [], 1.0)
        raise AssertionError('should have raised')
    except ValueError:
        print('case3 PASS')


def case_4():
    # round-trip with one-decimal-style ragged times (rounding residue absorption)
    words = [{'w': f'w{i}.', 'start': i * 0.37 + 0.03, 'end': i * 0.37 + 0.33}
             for i in range(7)]
    sents = [{'text': ' '.join(w['w'] for w in words[i:i + 2]), 'start': words[i]['start'],
              'end': words[min(i + 1, 6)]['end'], 'caption': f'cue{i}'}
             for i in range(0, 7, 2)]
    lines = bc.build_timed_script(sents, words, [], words[-1]['end'] + 0.11)
    script = '\n'.join(lines)
    bad = bc.verify_timed_script(script, ' '.join(w['w'] for w in words), words[-1]['end'] + 0.11)
    assert not bad, (bad, script)
    print('case4 PASS')


if __name__ == '__main__':
    case_1(); case_2(); case_3(); case_4()
    print('TIMED_UNIT_TESTS_ALL_PASS')
