#!/usr/bin/env python3
"""Regenerate docs/character-captions/ with the current stack.

The page was built 2026-07-21 and never regenerated since. Its procedural column
still shows the R_MIXD artefact ("very flat…") that the in-domain baseline fixed:
the published baseline gave Mixed Resonance a spread of 0.202 against a real spread
~3.9x larger, so a −0.9-unit offset became z = −4.6 and won the top-5 almost every time.

What is regenerated: the **procedural** column, the burst-span table, and the clip
header — all through `burst_captions.BurstCaptioner` at its current defaults
(locator model_v2.pt, classifier v2, 0.50/0.10/0.10, EmoNet on, in-domain baseline).

What is NOT regenerated: the **LLM-assisted** column. Those Gemma-4 rewordings are
carried over verbatim from the old page and labelled with their date, per the user's
decision. Two consequences are stated on the page rather than hidden:
  * the LLM column was reworded from the OLD procedural draft, so its wording still
    reflects the old baseline;
  * its burst positions came from the old locator run, so they can disagree with the
    newly detected spans shown in the table below each card.
"""
import html as H
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "docs", "character-captions")
OLD = os.path.join(OUT, "index.html")

CARD_RE = re.compile(r'<div class="card">.*?(?=<div class="card">|</body>|\Z)', re.S)
KEY_RE = re.compile(r'<span class="k">([^<]+)</span>')
MP3_RE = re.compile(r'src="audio/([^"]+)"')
LLM_RE = re.compile(r'<div><div class="h">(LLM-assisted[^<]*(?:<span[^>]*>[^<]*</span>)?[^<]*)</div>'
                    r'<div class="cap">(.*?)</div></div>', re.S)


def parse_old(path):
    """[{key, mp3, llm_head, llm_cap}] from the 2026-07-21 page."""
    s = open(path, encoding="utf-8").read()
    out = []
    for c in CARD_RE.findall(s):
        k = KEY_RE.search(c)
        m = MP3_RE.search(c)
        l = LLM_RE.search(c)
        if not (k and m):
            continue
        out.append({"key": k.group(1), "mp3": m.group(1),
                    "llm_head": l.group(1) if l else "", "llm_cap": l.group(2) if l else ""})
    return out


def esc(s):
    return H.escape(str(s), quote=True)


def cap_html(r) -> str:
    """Result dict -> the GENERAL/SCRIPT markup the old page used.

    Field names follow burst_captions.process(): `global_caption` and `script_lines`
    (each a dict with `cue` and `text`), matching how build_demo25.render_card reads
    them - not a `caption` string, which does not exist.
    """
    out = ["GENERAL: " + esc(r.get("global_caption", "")), "SCRIPT:"]
    for ln in r.get("script_lines", []):
        cue = esc(ln.get("cue", ""))
        txt = esc(ln.get("text", ""))
        # inline (bursts) already sit inside `text`; mark them like the old page did
        txt = re.sub(r'\(([^)]*)\)', lambda m: '<span class="c">(' + m.group(1) + ')</span>', txt)
        out.append((f'<span class="c">({cue})</span> ' if cue else "") + txt)
    return "<br>".join(out)


def burst_table(bursts):
    """`variant_a_bursts` -> the span table. Keys per build_demo25.render_card:
    kept / start / end / label / prob / p_noburst / dur / dur_floor."""
    if not bursts:
        return '<div class="none">no span above threshold</div>'
    rows = []
    for b in bursts:
        if b.get("kept"):
            cls, st = "st-kept", "kept"
            p = f'{b.get("prob",0):.2f}'
        elif b.get("label") is None:
            cls, st = "st-drop", "no_burst gate"
            p = "—"
        else:
            cls, st = "st-gate", f'gated &lt;{b.get("dur_floor",0):.2f}s'
            p = f'{b.get("prob",0):.2f}'
        rows.append(
            f'<tr><td>{b["start"]:.2f}&ndash;{b["end"]:.2f}s</td>'
            f'<td class="dur">{b.get("dur", b["end"]-b["start"]):.2f}s</td>'
            f'<td>{esc(b.get("label") or "—")}</td>'
            f'<td>{p}</td><td>{b.get("p_noburst",0):.2f}</td>'
            f'<td class="{cls}">{st}</td></tr>')
    return ('<table class="bt"><tr><th>span</th><th>dur</th><th>label</th>'
            '<th>p(class)</th><th>p(no_burst)</th><th>status</th></tr>'
            + "".join(rows) + "</table>")


def main():
    import burst_captions as B

    cards = parse_old(OLD)
    print(f"parsed {len(cards)} cards from the old page", flush=True)
    n_llm = sum(1 for c in cards if c["llm_cap"])
    print(f"  with an LLM column to carry over: {n_llm}", flush=True)
    if n_llm < len(cards):
        print(f"  WARNING: {len(cards)-n_llm} cards have no parseable LLM column", flush=True)

    bc = B.BurstCaptioner()
    t0 = time.time()
    for i, c in enumerate(cards):
        p = os.path.join(OUT, "audio", c["mp3"])
        if not os.path.exists(p):
            p = os.path.join(OUT, c["mp3"])
        try:
            r = bc.process(p, cid=c["key"])
        except Exception as e:
            print(f"  [{c['key']}] FAILED: {e}", flush=True)
            c["err"] = str(e)
            continue
        c["res"] = r
        print(f"  [{i+1}/{len(cards)}] {c['key']:18s} "
              f"{r.get('dur',0):5.1f}s  bursts={r.get('n_spans',0)}", flush=True)
    print(f"scored in {time.time()-t0:.0f}s", flush=True)

    ok = [c for c in cards if c.get("res")]
    json.dump([{k: v for k, v in c.items() if k != "res"} | {"result": c["res"]} for c in ok],
              open(os.path.join(OUT, "results_v2.json"), "w"), indent=1, default=str)

    body = []
    for c in ok:
        r = c["res"]
        gen = r.get("ei_gender")
        gtxt = f'<span style="color:#7cc7ff">EI gender {gen:+.2f}</span>' if gen is not None else ""
        body.append(
            '<div class="card"><div class="ch">'
            f'<span class="k">{esc(c["key"])}</span>\n'
            f'  <audio controls preload="none" src="audio/{esc(c["mp3"])}"></audio>'
            f'<span class="dd">clip {r.get("dur",0):.1f}s · {gtxt}</span></div>\n'
            '  <div class="col2">'
            '<div><div class="h">Procedural (tags, no archetype) '
            '<span class="fresh">regenerated</span></div>'
            f'<div class="cap">{cap_html(r)}</div></div>\n'
            f'  <div><div class="h">{c["llm_head"]} '
            '<span class="stale">from 2026-07-21 — not regenerated</span></div>'
            f'<div class="cap">{c["llm_cap"]}</div></div></div>\n'
            '  <div class="h" style="margin-top:9px">Detected burst spans '
            '(locator v2 → classifier v2, duration gate)</div>'
            + burst_table(r.get("variant_a_bursts", [])) + '</div>')

    old = open(OLD, encoding="utf-8").read()
    head = old.split('<div class="card">')[0]
    head = head.replace("</style>",
                        ".fresh{background:#1f6f3f;color:#d8ffe8;font-size:11px;padding:1px 6px;"
                        "border-radius:8px;margin-left:6px}\n"
                        ".stale{background:#5a4a1f;color:#ffe9b0;font-size:11px;padding:1px 6px;"
                        "border-radius:8px;margin-left:6px}\n</style>")
    note = (
        '<div class="note" style="border-left:3px solid #ffb454;padding:8px 12px;margin:10px 0">'
        '<b>Regenerated ' + time.strftime("%Y-%m-%d") + '.</b> The <b>procedural</b> column, the '
        'clip header and the burst-span table were rebuilt with the current stack: locator '
        '<code>model_v2.pt</code>, classifier <code>vocal-burst-detector-v2</code>, post-processing '
        '0.50&nbsp;/&nbsp;0.10&nbsp;/&nbsp;0.10, and the <b>in-domain baseline</b> that corrects the '
        'Mixed-Resonance scale error (published spread 0.202 vs a real spread ~3.9&times; larger, which '
        'turned a −0.9-unit offset into z&nbsp;=&nbsp;−4.6 and put "flat resonance" at the front of '
        'nearly every caption).<br>'
        'The <b>LLM-assisted</b> column is <b>unchanged from 2026-07-21</b>. It was reworded from the '
        '<i>old</i> procedural draft, so it still reflects the old baseline, and its burst positions '
        'came from the old locator — they can disagree with the spans in the table below each card. '
        'Compare the two columns as "new vs old", not as "procedural vs LLM".</div>')
    out = head + note + "\n".join(body) + "\n</body></html>\n"
    open(OLD, "w", encoding="utf-8").write(out)
    print(f"WROTE {OLD}  ({len(out)/1024:.0f} KB, {len(ok)} cards)", flush=True)


if __name__ == "__main__":
    main()
