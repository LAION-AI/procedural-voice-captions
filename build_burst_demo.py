#!/usr/bin/env python3
"""Build the burst-captions demo: run `burst_captions.BurstCaptioner` over a
diverse set of real clips (LAION character voices + a few in-the-wild demo
clips), write 120 kbps mono mp3s, and render docs/burst-captions/index.html
showing the global caption, per-sentence captions, and BOTH burst-insertion
variants side by side."""
import os, sys, glob, json, html
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import burst_captions as B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "docs", "burst-captions")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
DEMO_AUDIO = os.path.join(HERE, "docs", "audio")
ARCHEVO = "/run/user/1001/archevo/gen"
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- clip selection -------------------------------------------------------- #
# Character voices (one clip each) — screams, sobs, grunts, snarls, laughs.
CHARACTERS = ["pain_scream5", "pain_scream4", "zombie5", "ork5", "ranting5", "ranting4",
              "evil_ghost5", "goblin5", "goblin4", "mouse5", "dragon5", "fairy5",
              "sad_man5", "sad_woman5", "asmr_woman5", "asmr_man5"]
# In-the-wild demo clips (from the 100-clip grid) — bursty ones first.
DEMO_CLIPS = ["ZH_B00039_S04341_W000000", "ZH_B00039_S09711_W000002",
              "EN_B00015_S00125_W000000", "FR_B00000_S02278_W000000",
              "EN_B00000_S03298_W000123", "FR_B00001_S08689_W000031",
              "EN_B00026_S00187_W000083", "ZH_B00014_S01305_W000041"]


def pick_clips():
    clips = []
    for name in CHARACTERS:
        d = os.path.join(ARCHEVO, f"{name}_final64")
        wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
        if wavs:
            clips.append(("character", name, wavs[0]))
    for cid in DEMO_CLIPS:
        p = os.path.join(DEMO_AUDIO, cid + ".mp3")
        if os.path.exists(p):
            clips.append(("in-the-wild", cid, p))
    return clips


def esc(s):
    return html.escape(s or "")


def render(results):
    cards = []
    for kind, r in results:
        if "error" in r:
            cards.append(f'<div class="card"><b>{esc(r["id"])}</b> — ERROR: {esc(r["error"])}</div>')
            continue
        sid = r["id"]
        # per-sentence rows: base caption + variant B
        srows = []
        for s in r["sentences"]:
            burstB = (f' <span class="burst">({esc(s["variant_b_burst"])})</span>'
                      if s["variant_b_burst"] else ' <span class="none">— no burst</span>')
            srows.append(
                f'<div class="srow"><div class="stext">“{esc(s["text"]) or "…"}”</div>'
                f'<div class="scap">{esc(s["caption"])}</div>'
                f'<div class="scapB"><b>Variant B:</b> {esc(s["variant_b_caption"])}{burstB}</div></div>')
        # variant A: inline transcript + span table
        a_inline = esc(r["variant_a_inline"]) or "<i>(no words / no bursts)</i>"
        # bold the (bursts) inline
        import re
        a_inline = re.sub(r"\(([^)]+)\)", r'<span class="burst">(\1)</span>', a_inline)
        spans = []
        for b in r["variant_a_bursts"]:
            if b["kept"]:
                spans.append(f'<span class="sp keep">{b["start"]:.1f}–{b["end"]:.1f}s → '
                             f'{esc(b["label"])} ({b["prob"]:.2f})</span>')
            else:
                spans.append(f'<span class="sp disc">{b["start"]:.1f}–{b["end"]:.1f}s → '
                             f'discarded (P(no_burst)={b["p_noburst"]:.2f})</span>')
        spans_html = " ".join(spans) if spans else '<span class="sp">no span ≥ 0.7</span>'
        emo_note = "" if r["emonet"] else ' <span class="warn">(EmoNet off — VoiceNet+genuineness only)</span>'
        cards.append(f"""<div class="card">
  <div class="hd"><span class="tag {kind.replace(' ','-')}">{esc(kind)}</span>
    <span class="cid">{esc(sid)}</span><span class="meta">{r['dur']:.1f}s · genu {r['genu']} · blend {r['blend']}{emo_note}</span></div>
  <audio controls preload="none" src="audio/{esc(sid)}.mp3"></audio>
  <div class="asr"><b>ASR:</b> {esc(r['transcript']) or '<i>(no speech recognised)</i>'}</div>
  <div class="sec"><div class="lbl">Global caption</div><div class="gcap">{esc(r['global_caption'])}</div></div>
  <div class="sec"><div class="lbl">Per-sentence captions + Variant B (sentence-level bursts)</div>{''.join(srows) or '<i>(single segment)</i>'}</div>
  <div class="sec"><div class="lbl">Variant A — locator (precise inline bursts) &nbsp; <span class="k">{r['n_spans']} span(s) @0.7</span></div>
    <div class="ainline">{a_inline}</div>
    <div class="spans">{spans_html}</div></div>
</div>""")
    return "\n".join(cards)


CSS = """
body{background:#0f1117;color:#e6e9f0;font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;padding:18px 26px;max-width:1100px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 8px;color:#cfe7ff}
a{color:#7cc4ff}code{background:#1b1f2b;padding:1px 5px;border-radius:4px;font-size:12px}
.box{background:#12141b;border:1px solid #23283a;border-radius:10px;padding:14px 18px;margin:12px 0;color:#aab3c5}.box b{color:#cfe7ff}
.rec{border-color:#3a5;background:#10160f}.rec b{color:#8fe6a0}
.card{background:#161922;border:1px solid #262b3a;border-radius:10px;padding:12px 15px;margin:12px 0}
.hd{display:flex;align-items:center;gap:9px;margin-bottom:7px;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
.character{background:rgba(255,150,90,.18);color:#ffb27a}.in-the-wild{background:rgba(120,180,255,.18);color:#8fbaff}
.cid{font-weight:600;color:#e6e9f0}.meta{color:#7f889c;font-size:12px}.warn{color:#e0b062}
audio{height:32px;width:340px;margin:2px 0}
.asr{color:#9aa3b5;font-size:12.5px;margin:6px 0 2px}
.sec{margin-top:9px}.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#6f7a90;margin-bottom:4px}.k{color:#7f889c;text-transform:none;letter-spacing:0}
.gcap{background:#11141c;border-left:3px solid #4a6cff;padding:6px 10px;border-radius:5px}
.srow{border-left:2px solid #2a3040;padding:4px 0 4px 10px;margin:5px 0}
.stext{color:#c3ccdd;font-style:italic}.scap{color:#9aa3b5;font-size:13px}.scapB{color:#c6cfe0;font-size:13px;margin-top:2px}
.ainline{background:#11141c;border-left:3px solid #e0873a;padding:7px 10px;border-radius:5px;line-height:1.7}
.burst{color:#ffb27a;font-weight:700}.none{color:#5f6878;font-size:12px}
.spans{margin-top:5px}.sp{display:inline-block;background:#1c2130;padding:2px 7px;border-radius:5px;margin:2px;font-size:11.5px;color:#aeb6c8}
.sp.keep{background:rgba(255,150,90,.16);color:#ffb27a;font-weight:600}.sp.disc{color:#7f889c;font-style:italic}
"""


def main():
    use_emonet = os.environ.get("BC_EMONET", "1") != "0"
    bc = B.BurstCaptioner(device=os.environ.get("BC_DEVICE", "cuda:0"), use_emonet=use_emonet)
    clips = pick_clips()
    print(f"selected {len(clips)} clips")
    results = []
    for i, (kind, cid, path) in enumerate(clips):
        mp3 = os.path.join(AUDIO_DIR, cid + ".mp3")
        try:
            r = bc.process(path, cid=cid, mp3_out=mp3)
        except Exception as e:
            r = {"id": cid, "error": repr(e)}
        results.append((kind, r))
        print(f"  [{i+1}/{len(clips)}] {cid}: {r.get('global_caption','ERR')[:80]}", flush=True)

    json.dump({"n": len(results), "results": [r for _, r in results]},
              open(os.path.join(OUT_DIR, "results.json"), "w"), ensure_ascii=False, indent=1)

    body = render(results)
    n_char = sum(1 for k, _ in results if k == "character")
    n_wild = len(results) - n_char
    emo_state = "on (Empathic-Insight-Voice-Plus)" if use_emonet else "OFF — VoiceNet + genuineness only"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Procedural Voice Captions — with detected vocal bursts</title><style>{CSS}</style></head><body>
<h1>Procedural voice captions <span style="color:#ffb27a">with detected vocal bursts</span></h1>
<p style="color:#8b93a7;margin-top:2px">{n_char} LAION character voices + {n_wild} in-the-wild clips ·
<a href="../">← back to the caption grid</a></p>

<div class="box"><p><b>What this page adds.</b> The base <a href="../">procedural captioner</a> turns
model predictions into a voice description. Here we additionally <b>detect the non-verbal vocal bursts</b>
(laughs, gasps, sighs, screams, sobs, grunts, slaps…) that the audio really contains and <b>write them into
the captions</b>. Every number below is produced by models listening to the clip — nothing is hand-written.</p>
<p><b>Scoring stack (all run on the audio):</b> VoiceNet 57-dim predictors + genuineness + vocal-burst-blend
(VoiceCLAP-commercial embedding → per-dim MLP heads); EmoNet-40 ({emo_state});
Parakeet-TDT for word + sentence timestamps; the <code>vocalburst-locator</code> (50 fps burst probability)
and a VoiceCLAP→MLP multi-label burst classifier (82 taxonomy classes + <code>no_burst</code>).</p></div>

<div class="box"><p><b>Caption composition.</b><br>
• <b>Global caption</b> = Top-5 VoiceNet dims + Top-3 EmoNet emotions + genuineness + Age + Gender + Tempo.
Bursts are <i>not</i> placed at global level.<br>
• <b>Per-sentence caption</b> = Top-3 VoiceNet dims + Top-3 emotions (no Age/Gender/Tempo — those are global only).</p>
<p><b>Two burst-insertion variants (compare them per clip below):</b><br>
<span class="burst">Variant A — locator (precise position).</span> Scan the whole clip with the burst locator at
threshold <b>0.7</b>; each detected time-span's audio → classifier → if <code>P(no_burst) &lt; 0.5</code> take the
<b>top-1</b> class and insert it as its own <span class="burst">(Class)</span> inline at that moment, between the
two ASR words nearest the burst time. Otherwise the span is discarded (no false alarm).<br>
<span class="burst">Variant B — sentence-level.</span> Run the classifier on each whole sentence segment; if
<code>P(no_burst) ≥ 0.5</code> attach nothing, else weave the top-1 class into that sentence's caption with a
small procedurally-generated phrase (<i>“… punctuated by a (Gasp).”</i>).</p></div>

<div class="box rec"><p><b>Recommendation: use Variant A (locator) as the primary, Variant B as a fallback.</b>
Variant A places the burst <i>where it actually happens</i> — the inline transcript reads like a real script
(<i>“(Surprised Gasp) No, no, (Surprised Gasp) oh, it hurts.”</i>), which is exactly the per-event, time-anchored
signal the voice-acting format wants, and its explicit <code>P(no_burst)&lt;0.5</code> gate suppresses false alarms
per-span. Variant B is coarser: it can only say a sentence <i>contains</i> some burst, it collapses multiple bursts
in one sentence into a single label, and running the classifier over a whole sentence dilutes a short burst among
seconds of speech (lower recall on brief events). <b>Prefer Variant B only when</b> word-level timestamps are missing
(non-speech screams, noisy ASR) or when a compact one-label-per-sentence summary is all that's needed — there,
Variant A has nowhere to anchor and Variant B still adds useful information.</p></div>

<h2>Clips</h2>
{body}
<p style="color:#6f7a90;margin-top:24px;font-size:12px">Generated by <code>build_burst_demo.py</code> ·
pipeline in <code>burst_captions.py</code>. Bursts, emotions, dimensions and timestamps are all model predictions.</p>
</body></html>"""
    open(os.path.join(OUT_DIR, "index.html"), "w").write(doc)
    print("wrote", os.path.join(OUT_DIR, "index.html"))


if __name__ == "__main__":
    main()
