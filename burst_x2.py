#!/usr/bin/env python3
r"""The **x2 vocal-burst classifier** — a second, narrower opinion on a located span.

`laion/vocal-burst-detector-x2` names a burst over **17 classes = 16 bursts +
`no_burst`**, where the shipped `laion/vocal-burst-detector-v2` names it over 83
(82 taxonomy classes + `no_burst`, 77 reachable). It is *not* a replacement, and this
module is written so that it cannot become one by accident:

**The 16 x2 classes are a strict subset of the repo's 82-class taxonomy.** Every one of
them already appears in `vocalburst_taxonomy.json`; none is new. So the x2 head adds no
vocabulary — it removes 66 classes. Seven of the taxonomy's sixteen groups
(`crying_and_distress`, `eating_and_drinking`, `hand_and_body_sounds`,
`mouth_and_lip_sounds`, `throat_and_vocal_sounds`, `tongue_clicks`, `whistling`) become
unsayable. On 12,894 burst events of the annotated LAION-TTS corpus, **83.8 % carry a
label the x2 head cannot emit** — `Low Mumble` 26.3 %, `Ahem` 26.2 %, `Contented Sigh`
20.5 %, `Surprised Gasp` 7.2 %. Switching the pipeline over would silently delete the
four most common non-verbal sounds from the captions.

What the x2 head *is* good for is the 16 classes it does know, where it was measured
better than anything else this project has: 0.7536 argmax on held-out real audio
(0.8480 at family level) against 0.0588 chance, with the 3584-d `laion/voiceclap-large-v2`
encoder. So it is wired in as an **extra naming source with an explicit jurisdiction**,
and every burst record says which vocabulary produced it.

Three things this module provides
---------------------------------

1. **`RecallTable`** — the per-class recall floor that ships with the detector
   (`per_class_recall.json`, bundled here as `vocalburst_x2_recall*.json`), plus the
   **confidence tiering** built on top of it. Pure stdlib: no torch, no network.
2. **`merge_label()`** — the policy that decides which vocabulary writes the caption.
3. **`X2Classifier`** / **`VoiceCLAPLargeV2`** — the model half, imported lazily.

The confidence tiers, and why
-----------------------------

The detector's own README says it plainly: *"a class recalled at 20 % cannot show a hit
rate meaningfully above 20 %, however good the generator is. Put the recall floor next to
every row."* The same is true of a caption. Before this module, the repo asserted every
burst class with identical certainty — `(Sharp Inhale)` (recalled 0.897 on real audio) and
`(Relief Sigh)` (recalled 0.062) were written the same way. That is a claim the numbers do
not support.

So a class is written at the strength its measured recall earns:

| tier | condition (recall on **real** audio) | written as | example |
|---|---|---|---|
| `named` | strict ≥ `BURST_X2_STRICT_MIN` (0.50) | the class name | `(Chuckle)` |
| `hedged` | strict ≥ `BURST_X2_HEDGE_MIN` (0.25) | class name + `?` | `(Breathy Giggle?)` |
| `family` | strict below that, but **family** recall ≥ `BURST_X2_FAMILY_MIN` (0.50) | the family word | `(Breath)` |
| `generic` | nothing clears | `BURST_X2_GENERIC_LABEL` | `(Vocal Burst)` |

The `family` tier is the interesting one and it is not a fudge: the detector confuses
`Deep Breath` with `Heavy Breathing` far more often than it confuses either with a laugh
(strict recall 0.186, family recall 0.712). Naming the family is a *true* statement where
naming the class is a coin flip. The `generic` tier keeps only what the binary
burst/no-burst discriminator supports on its own — 98.6 % accurate for `large-v2` — which
is a great deal more than nothing.

With the bundled `large-v2` table and the default thresholds that is 6 classes `named`,
5 `hedged`, 2 `family`, 3 `generic`. All four thresholds are env-tunable, and
`BURST_X2_RECALL_SOURCE` chooses which measurement they read (`real`, the in-the-wild
held-out set, is the default and the conservative one; `dramabox` is the in-domain set;
`min` takes the worse of the two).

Choosing a head
---------------

| `BURST_X2_HEAD` | encoder | dim | val_acc | real all | real family | load |
|---|---|---:|---:|---:|---:|---:|
| `large-v2` (default) | `laion/voiceclap-large-v2` | 3584 | 0.7056 | 0.7536 | 0.8480 | ~101 s |
| `commercial` | `laion/voiceclap-commercial` | 768 | 0.6258 | 0.7039 | 0.7954 | ~0 s |

`large-v2` wins every cell of that table, so it is the default. `commercial` is the
cheap alternative and it is cheap in a way that is specific to *this* repo: the
`laion/voiceclap-commercial` encoder is **already loaded** by `BurstCaptioner` for the
VoiceNet heads and the v2 classifier, so the commercial x2 head costs five
768→256→17 MLPs — about 1 MB — and no extra encoder at all. `large-v2` brings a second,
7B-parameter tower with it.

(`voiceclap-small-v2` scores between the two but is **CC BY-NC 4.0** and is deliberately
not wired in here.)

One correctness note that is easy to get wrong: both x2 heads were trained on
**L2-normalised** embeddings, and the commercial encoder was called with an explicit
`sample_rate=`. The repo's own v2 classifier uses the *un*-normalised output of the same
encoder. So the commercial x2 head cannot simply reuse `BurstCaptioner.classify_burst`'s
embedding — `embed_commercial()` below re-normalises. Feeding it the raw embedding
loads cleanly and quietly produces nonsense.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# ENV
X2_HEAD_NAME     = os.environ.get("BURST_X2_HEAD", "large-v2")     # large-v2 | commercial
X2_REPO          = os.environ.get("BURST_X2_REPO", "laion/vocal-burst-detector-x2")
X2_CKPT_DIR      = os.environ.get("BURST_X2_CKPT_DIR", "")         # local dir with the 5 .pt files
VOICECLAP_LARGE  = os.environ.get("VOICECLAP_LARGE_REPO", "laion/voiceclap-large-v2")
X2_CACHE         = os.environ.get("BURST_X2_CACHE", "")            # writable dir for the remapped LoRA

# Which measured recall the tiers read. `real` = 992-clip in-the-wild held-out set (the honest,
# cross-domain one, and the default); `dramabox` = the in-domain set; `min` = the worse of the two.
RECALL_SOURCE    = os.environ.get("BURST_X2_RECALL_SOURCE", "real")
STRICT_MIN       = float(os.environ.get("BURST_X2_STRICT_MIN", "0.50"))
HEDGE_MIN        = float(os.environ.get("BURST_X2_HEDGE_MIN", "0.25"))
FAMILY_MIN       = float(os.environ.get("BURST_X2_FAMILY_MIN", "0.50"))
HEDGE_MARK       = os.environ.get("BURST_X2_HEDGE_MARK", "?")
GENERIC_LABEL    = os.environ.get("BURST_X2_GENERIC_LABEL", "Vocal Burst")

# Which vocabulary writes the caption. See `merge_label`.
LABEL_SOURCE     = os.environ.get("BURST_LABEL_SOURCE", "v2")      # v2 | x2 | union
# Optional SECOND no_burst gate from the x2 head. Off by default on purpose: the v2 gate has
# already decided which spans survive, and letting a narrower head veto spans it has no
# vocabulary for would delete bursts rather than name them. Set to a probability to enable.
X2_GATE          = float(os.environ.get("BURST_X2_GATE", "0") or 0)

SR = 16000
BUNDLED_RECALL = {
    "large-v2":   os.path.join(HERE, "vocalburst_x2_recall.json"),
    "commercial": os.path.join(HERE, "vocalburst_x2_recall_commercial.json"),
}
# Sub-directory of `laion/vocal-burst-detector-x2` holding each head's five checkpoints.
HEAD_SUBDIR = {"large-v2": "production", "commercial": "commercial"}
HEAD_DIM = {"large-v2": 3584, "commercial": 768}
# The encoder each head was trained on. `commercial` is the one `BurstCaptioner` already loads.
HEAD_ENCODER_HF = {"large-v2": VOICECLAP_LARGE,
                   "commercial": os.environ.get("VOICECLAP_REPO", "laion/voiceclap-commercial")}
CKPT_GLOB = "vocal_burst_mlp_prod_s{}.pt"
N_MEMBERS = 5

TIERS = ("named", "hedged", "family", "generic")


# --------------------------------------------------------------------------- #
# The recall floor and the tiers built on it — stdlib only, so this half is testable
# and usable with no models, no torch and no network.
class RecallTable:
    """`per_class_recall.json` + the confidence tiering.

    ``strict(label)`` is the probability that the detector named *this exact class* on
    held-out audio; ``family(label)`` that it named *some class in the same family*.
    ``tier(label)`` turns the pair into one of ``named`` / ``hedged`` / ``family`` /
    ``generic``, and ``write(label)`` into the text a caption may honestly carry.
    """

    def __init__(self, data, source=None, strict_min=None, hedge_min=None, family_min=None,
                 hedge_mark=None, generic_label=None):
        self.data = data
        self.classes = list(data["classes"])
        self.bursts = [c for c in self.classes if c != "no_burst"]
        self.family_of = dict(data["family_of"])
        self.encoder = data.get("encoder")
        self.chance = float(data.get("chance", 1.0 / max(len(self.classes), 1)))
        self.source = source or RECALL_SOURCE
        self.strict_min = STRICT_MIN if strict_min is None else float(strict_min)
        self.hedge_min = HEDGE_MIN if hedge_min is None else float(hedge_min)
        self.family_min = FAMILY_MIN if family_min is None else float(family_min)
        self.hedge_mark = HEDGE_MARK if hedge_mark is None else hedge_mark
        self.generic_label = GENERIC_LABEL if generic_label is None else generic_label

    # -- loading ------------------------------------------------------------- #
    @classmethod
    def load(cls, head=None, path=None, **kw):
        """Bundled table for `head` ('large-v2' / 'commercial'), or an explicit `path`."""
        if path is None:
            head = head or X2_HEAD_NAME
            path = BUNDLED_RECALL.get(head)
            if path is None:
                raise ValueError(f"unknown x2 head {head!r}; know {sorted(BUNDLED_RECALL)}")
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), **kw)

    # -- the numbers --------------------------------------------------------- #
    def _cell(self, label, key):
        rec = self.data.get("recall", {}).get(label)
        if not rec:
            return None
        if self.source == "min":
            vals = [v.get(key) for v in rec.values() if isinstance(v, dict) and v.get(key) is not None]
            return min(vals) if vals else None
        v = rec.get(self.source)
        return v.get(key) if isinstance(v, dict) else None

    def strict(self, label):
        """Recall of this exact class, or None when the label is outside the x2 vocabulary."""
        v = self._cell(label, "recall")
        return None if v is None else float(v)

    def family_recall(self, label):
        """Recall of *some* class in the same family, or None."""
        v = self._cell(label, "family_recall")
        return None if v is None else float(v)

    def n_heldout(self, label):
        v = self._cell(label, "n_heldout")
        return None if v is None else int(v)

    def reliable(self, label):
        """Whether the held-out count clears the table's own `min_reliable_n`."""
        return bool(self._cell(label, "reliable"))

    def family(self, label):
        return self.family_of.get(label)

    def in_vocab(self, label):
        return label in self.bursts

    # -- the tiering --------------------------------------------------------- #
    def tier(self, label):
        """'named' | 'hedged' | 'family' | 'generic' — how strongly `label` may be asserted."""
        s = self.strict(label)
        if s is None:
            return "generic"
        if s >= self.strict_min:
            return "named"
        if s >= self.hedge_min:
            return "hedged"
        f = self.family_recall(label)
        if f is not None and f >= self.family_min and self.family(label) not in (None, "no_burst"):
            return "family"
        return "generic"

    def family_word(self, label):
        """The family rendered as a caption word: 'breath' -> 'Breath'."""
        fam = self.family(label)
        if not fam or fam == "no_burst":
            return self.generic_label
        return " ".join(w.capitalize() for w in str(fam).replace("_", " ").split())

    def write(self, label, tier=None):
        """The text a caption may carry for `label`, at its earned certainty."""
        t = tier or self.tier(label)
        if t == "named":
            return label
        if t == "hedged":
            return f"{label}{self.hedge_mark}"
        if t == "family":
            return self.family_word(label)
        return self.generic_label

    def describe(self, label):
        """Everything the caption record should carry about one x2 label."""
        t = self.tier(label)
        return {"tier": t, "written": self.write(label, t),
                "recall": self.strict(label), "family_recall": self.family_recall(label),
                "family": self.family(label), "recall_source": self.source,
                "n_heldout": self.n_heldout(label), "recall_reliable": self.reliable(label)}

    def summary(self):
        """{tier: [class, ...]} for the whole vocabulary — what the thresholds actually do."""
        out = {t: [] for t in TIERS}
        for c in self.bursts:
            out[self.tier(c)].append(c)
        return out


# --------------------------------------------------------------------------- #
# Which vocabulary writes the caption.
def merge_label(v2_label, x2, table, policy=None):
    """Decide the label a span is written with, and record where it came from.

    `v2_label`  — what `laion/vocal-burst-detector-v2` named (None if its gate fired).
    `x2`        — one dict from `X2Classifier.classify()`, or None if x2 did not run.
    `table`     — a `RecallTable` for the head that produced `x2`.
    `policy`    — `v2` | `x2` | `union` (default `$BURST_LABEL_SOURCE`, itself `v2`).

    Returns ``{"label", "written", "vocab", "tier", "source", ...}`` where **`written` is
    what goes into the caption** and **`vocab` says which vocabulary earned the right to
    say it**: ``v2-83`` (the 82-class taxonomy), ``x2-17`` (the 16-class head, exact
    class), ``x2-17-family`` (backed off to the family) or ``generic``.

    The three policies:

    * **`v2`** — the shipped behaviour, byte for byte. x2 is recorded next to the span and
      changes nothing. This is the default, so enabling the x2 head never silently
      rewrites a caption.
    * **`x2`** — the x2 head names every span, at its earned certainty. Honest, and much
      poorer: on the corpus this repo captions, 83.8 % of burst events carry a label x2
      cannot emit, and those become family words or `Vocal Burst`.
    * **`union`** — x2 names the span **only where it has jurisdiction**, i.e. where the
      v2 label is one of x2's 16 classes and both heads are talking about the same thing.
      Where the v2 label is outside that set — `Ahem`, `Low Mumble`, `Contented Sigh`,
      `Surprised Gasp`, everything in seven whole taxonomy groups — the v2 label stands
      untouched and is flagged `out_of_x2_vocab`. This keeps the coverage of the old
      vocabulary and takes the accuracy of the new one exactly where it applies.
    """
    pol = (policy or LABEL_SOURCE or "v2").lower()
    if pol not in ("v2", "x2", "union"):
        raise ValueError(f"BURST_LABEL_SOURCE must be v2|x2|union, got {pol!r}")

    base = {"label": v2_label, "written": v2_label, "vocab": "v2-83", "tier": "named",
            "source": "v2", "policy": pol, "out_of_x2_vocab": None}
    if v2_label is None:
        # The v2 `no_burst` gate rejected this span, and that is the last word under every
        # policy. The x2 head names bursts; it does not vote on whether they exist. (A caller
        # that really wants a second existence check sets BURST_X2_GATE, which only ever
        # removes spans, and records the removal.)
        base.update({"written": None, "vocab": None, "tier": None, "source": None})
        return base
    if pol == "v2" or x2 is None or not x2.get("label"):
        return base
    if x2["label"] == "no_burst":                 # x2 has no opinion worth writing
        return base

    in_vocab = table.in_vocab(v2_label) if v2_label else False
    base["out_of_x2_vocab"] = (v2_label is not None and not in_vocab)
    if pol == "union" and not in_vocab:
        # The narrower head has nothing to say here. Keep the wider vocabulary's word.
        return base

    d = table.describe(x2["label"])
    vocab = {"named": "x2-17", "hedged": "x2-17", "family": "x2-17-family",
             "generic": "generic"}[d["tier"]]
    base.update({"label": x2["label"], "written": d["written"], "vocab": vocab,
                 "tier": d["tier"], "source": "x2", "x2_recall": d["recall"],
                 "x2_family_recall": d["family_recall"], "x2_family": d["family"],
                 "recall_source": d["recall_source"], "recall_reliable": d["recall_reliable"],
                 "agrees_with_v2": (v2_label == x2["label"]) if v2_label else None})
    return base


# --------------------------------------------------------------------------- #
# Variant-B phrasing, at the tier's certainty.
PHRASES_NAMED = ["punctuated by a ({b})", "with an audible ({b})", "broken by a ({b})",
                 "carrying a ({b})", "interrupted by a ({b})", "marked by a ({b})"]
PHRASES_HEDGED = ["possibly punctuated by a ({b})", "with what sounds like a ({b})",
                  "apparently broken by a ({b})", "seemingly carrying a ({b})",
                  "with a probable ({b})", "marked by what may be a ({b})"]


def phrases_for(tier):
    """The Variant-B templates a tier may use.

    `hedged` gets its own set, because a bare `?` inside a prose cue reads as a typo.
    `family` and `generic` use the plain templates — their hedging is already in the
    label itself (`(Breath)`, `(Vocal Burst)`), and doubling it would be mush.
    """
    return PHRASES_HEDGED if tier == "hedged" else PHRASES_NAMED


# --------------------------------------------------------------------------- #
# The model half. Imported lazily so everything above stays stdlib-only.
def _snapshot(repo, allow_patterns=None):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=allow_patterns)


class VoiceCLAPLargeV2:
    """`laion/voiceclap-large-v2` (3584-d) without sentence-transformers.

    The published repo is a sentence-transformers bundle: Transformer -> Pooling
    (`lasttoken`) -> Normalize, over a Qwen2.5-Omni thinker with a LoRA adapter. Rather
    than take the sentence-transformers dependency, the four steps are reproduced from
    the repo's own config files — `modules.json`, `1_Pooling/config.json` and the
    `sentence_transformers.jinja` chat template — so nothing here is guessed.

    Two traps, both load-bearing:

    * **The adapter keys carry one extra `model.` level**, because the LoRA was trained
      through sentence-transformers, whose Transformer module holds the HF model in an
      attribute called `model`. PEFT does not complain about keys it cannot place: it
      loads cleanly and leaves the base model **completely unchanged**, which reads
      downstream as "the encoder is no better than chance" rather than as a bug. So the
      adapter is remapped and the match is asserted, not assumed.
    * **The last token is taken from the attention mask**, not `[:, -1]`. The processor
      pads on the right, so `[:, -1]` would pool a pad token for every sequence shorter
      than the longest in its batch.
    """

    def __init__(self, path=None, device=None, dtype=None, cache_dir=None):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        path = path or VOICECLAP_LARGE
        if not os.path.isdir(path):
            path = _snapshot(path)
        self.path = path
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.proc = Qwen2_5OmniProcessor.from_pretrained(path)
        with open(os.path.join(path, "additional_chat_templates",
                               "sentence_transformers.jinja"), encoding="utf-8") as f:
            self.tpl = f.read()
        with open(os.path.join(path, "1_Pooling", "config.json"), encoding="utf-8") as f:
            pool = json.load(f)
        assert pool["pooling_mode"] == "lasttoken", f"unexpected pooling {pool['pooling_mode']}"
        self.dim = int(pool["embedding_dimension"])
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            path, dtype=(dtype or torch.bfloat16)).to(self.dev).eval()
        self._attach_adapter(cache_dir)

    def _attach_adapter(self, cache_dir):
        import shutil
        import tempfile
        from peft import PeftModel
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        src = os.path.join(self.path, "adapter")
        out = cache_dir or X2_CACHE or os.path.join(tempfile.gettempdir(),
                                                    "pvc_voiceclap_large_v2_adapter")
        stamp = os.path.join(out, ".remapped_ok")
        if not os.path.exists(stamp):
            os.makedirs(out, exist_ok=True)
            sd = load_file(os.path.join(src, "adapter_model.safetensors"))
            fixed, n = {}, 0
            for k, v in sd.items():
                nk = k.replace("base_model.model.model.", "base_model.model.", 1)
                n += int(nk != k)
                fixed[nk] = v
            save_file(fixed, os.path.join(out, "adapter_model.safetensors"))
            shutil.copy2(os.path.join(src, "adapter_config.json"),
                         os.path.join(out, "adapter_config.json"))
            with open(stamp, "w", encoding="utf-8") as f:
                f.write(f"{len(sd)} tensors, {n} keys remapped\n")
        with safe_open(os.path.join(out, "adapter_model.safetensors"), "pt") as f:
            want = set(f.keys())
        self.model = PeftModel.from_pretrained(self.model, out, is_trainable=False)
        have = {k.replace(".default", "") for k in self.model.state_dict() if "lora_" in k}
        missing = sorted(want - have)
        assert not missing, (f"{len(missing)} of {len(want)} adapter tensors did not match the "
                             f"model — the encoder would be silently unadapted, e.g. {missing[:3]}")
        self.n_lora = len(want)
        self.model.eval()

    def _texts(self, n):
        msgs = [{"role": "user", "content": [{"type": "audio", "audio": None}]}]
        one = self.proc.apply_chat_template(msgs, chat_template=self.tpl, tokenize=False,
                                            add_generation_prompt=True)
        if isinstance(one, list):
            one = one[0]
        return [one] * n

    def encode(self, wavs16):
        """[1-D float32 @16 kHz] -> (N, 3584) float32, L2-normalised."""
        import numpy as np
        import torch
        wavs = [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs16]
        wavs = [w if len(w) >= 400 else np.pad(w, (0, 400 - len(w))) for w in wavs]
        b = self.proc(text=self._texts(len(wavs)), audio=wavs, sampling_rate=SR,
                      return_tensors="pt", padding=True)
        b = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in b.items()}
        with torch.no_grad():
            out = self.model(**b, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[-1]
            am = b["attention_mask"]
            idx = am.sum(1) - 1
            if am[0, 0].item() == 0:                       # left-padded
                idx = torch.full_like(idx, h.shape[1] - 1)
            e = h[torch.arange(h.shape[0], device=h.device), idx]
            e = torch.nn.functional.normalize(e.float(), p=2, dim=-1)
        return e.cpu().numpy().astype("float32")


class X2Classifier:
    """The 5-member x2 ensemble: encoder -> 5 x (D -> 256 -> 17) -> mean softmax.

    `embed_fn` lets the caller hand in an encoder it has already paid for. That is the
    whole point of the `commercial` head here: `BurstCaptioner` loads
    `laion/voiceclap-commercial` anyway, so the x2 commercial head costs five small MLPs
    and nothing else. Left as None, the head loads its own encoder — for `large-v2` that
    is a second 7B tower and about 101 s.

    `classify()` mirrors the detector's own `ProductionBurstScorer.classify`: it **drops
    nothing**. Gating is the caller's business, and by default this repo does not let x2
    gate at all (see `merge_label`).
    """

    def __init__(self, head=None, device=None, ckpt_dir=None, embed_fn=None, verbose=False):
        import torch
        import torch.nn as nn
        self.head = head or X2_HEAD_NAME
        if self.head not in HEAD_SUBDIR:
            raise ValueError(f"unknown x2 head {self.head!r}; know {sorted(HEAD_SUBDIR)}")
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        paths = self._resolve_ckpts(ckpt_dir)
        self.members = []
        ck = None
        for p in paths:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            a = ck["arch"]
            m = nn.Sequential(nn.Linear(a["D"], a["H"]), nn.BatchNorm1d(a["H"]), nn.GELU(),
                              nn.Dropout(a.get("dropout", 0.3)), nn.Linear(a["H"], a["C"]))
            m.load_state_dict({k.replace("net.", ""): v for k, v in ck["state_dict"].items()})
            self.members.append(m.to(self.dev).eval())
        self.classes = list(ck["classes"])
        self.family = dict(zip(ck["classes"], ck.get("family") or ck["classes"]))
        self.dim = int(ck["arch"]["D"])
        self.nb_ix = self.classes.index("no_burst")
        expect = HEAD_DIM[self.head]
        assert self.dim == expect, f"{self.head} head is {self.dim}-d, expected {expect}"
        self.table = RecallTable.load(self.head)
        assert self.table.classes == self.classes, "recall table and checkpoint disagree on classes"
        self._embed = embed_fn or self._own_encoder()

    def _resolve_ckpts(self, ckpt_dir):
        import glob as _glob
        d = ckpt_dir or X2_CKPT_DIR
        if d:
            paths = sorted(_glob.glob(os.path.join(d, CKPT_GLOB.format("*"))))
            if not paths:
                raise FileNotFoundError(f"no {CKPT_GLOB.format('*')} under {d}")
            return paths[:N_MEMBERS]
        sub = HEAD_SUBDIR[self.head]
        root = _snapshot(X2_REPO, allow_patterns=[f"{sub}/*.pt", f"{sub}/*.json"])
        paths = sorted(_glob.glob(os.path.join(root, sub, CKPT_GLOB.format("*"))))
        if not paths:
            raise FileNotFoundError(f"no {sub}/{CKPT_GLOB.format('*')} in {X2_REPO}")
        return paths[:N_MEMBERS]

    def _own_encoder(self):
        if self.head == "large-v2":
            enc = VoiceCLAPLargeV2(device=self.dev)
            assert enc.dim == self.dim, f"encoder dim {enc.dim} != head dim {self.dim}"
            return enc.encode
        from transformers import AutoModel
        vc = AutoModel.from_pretrained(HEAD_ENCODER_HF[self.head],
                                       trust_remote_code=True).to(self.dev).eval()
        return lambda w: embed_commercial(vc, w, self.dev)

    # -- naming -------------------------------------------------------------- #
    def probs(self, cuts16, batch_samples=24 * SR * 4):
        """[1-D float32 @16 kHz] -> (N, 17) mean softmax over the five members."""
        import numpy as np
        import torch
        if not cuts16:
            return np.zeros((0, len(self.classes)), dtype="float32")
        cuts = [np.asarray(_np1d(c), dtype=np.float32).reshape(-1) for c in cuts16]
        cuts = [c if len(c) >= 800 else np.pad(c, (0, 800 - len(c))) for c in cuts]
        out = np.zeros((len(cuts), len(self.classes)), dtype="float32")

        def flush(buf):
            if not buf:
                return
            E = torch.as_tensor(self._embed([cuts[i] for i in buf])).to(self.dev).float()
            if E.dim() == 1:
                E = E.unsqueeze(0)
            with torch.no_grad():
                pr = torch.zeros(len(buf), len(self.classes), device=self.dev)
                for m in self.members:
                    pr += torch.softmax(m(E), -1)
                pr = (pr / len(self.members)).cpu().numpy()
            for j, i in enumerate(buf):
                out[i] = pr[j]

        buf = []
        for i in sorted(range(len(cuts)), key=lambda i: len(cuts[i])):   # length-sorted batching
            if buf and (len(buf) + 1) * max(len(cuts[i]), 800) > batch_samples:
                flush(buf)
                buf = []
            buf.append(i)
        flush(buf)
        return out

    def classify(self, cuts16, topk=3, mask_no_burst=True):
        """Name spans somebody else cut. Nothing is dropped; apply your own gate.

        With `mask_no_burst` (the default here, and the difference from the detector's own
        `classify`) the `no_burst` class is excluded from the argmax, because in this repo
        the v2 gate has *already* decided that the span is a burst. Its probability is
        still reported as `no_burst_prob` so a caller can gate on it if it wants to.
        """
        import numpy as np
        P = self.probs(cuts16)
        res = []
        for row in P:
            r = row.copy()
            if mask_no_burst:
                r[self.nb_ix] = -1.0
            o = np.argsort(-r)
            k = int(o[0])
            top = [(self.classes[j], float(row[j])) for j in o[:topk]]
            lab = self.classes[k]
            d = self.table.describe(lab)
            res.append({"label": lab, "prob": float(row[k]), "labels": top,
                        "family": self.family.get(lab), "no_burst_prob": float(row[self.nb_ix]),
                        "tier": d["tier"], "written": d["written"], "recall": d["recall"],
                        "family_recall": d["family_recall"]})
        return res


def embed_commercial(vc, wavs16, device):
    """`laion/voiceclap-commercial` -> (N, 768) float32, **L2-normalised**.

    The normalisation is not optional and it is not what the repo's v2 classifier does:
    the x2 commercial head was trained on normalised embeddings, and feeding it raw ones
    produces a confident, wrong answer with no error anywhere.
    """
    import numpy as np
    import torch
    wavs = [np.asarray(_np1d(w), dtype=np.float32).reshape(-1) for w in wavs16]
    n = max(400, max(len(w) for w in wavs))
    x = np.zeros((len(wavs), n), dtype=np.float32)
    for i, w in enumerate(wavs):
        x[i, :len(w)] = w
    with torch.no_grad():
        e = vc.encode_waveform(torch.from_numpy(x).to(device), sample_rate=SR)
        e = torch.nn.functional.normalize(e.float(), p=2, dim=-1)
    return e.cpu().numpy().astype("float32")


def _np1d(x):
    """Accept a torch tensor, a numpy array or a list."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x


# --------------------------------------------------------------------------- #
def main():
    """`python burst_x2.py` — print what the thresholds do to each head's vocabulary."""
    import argparse
    ap = argparse.ArgumentParser(description="x2 recall floor and confidence tiers")
    ap.add_argument("--head", default=None, help="large-v2 | commercial | both (default both)")
    ap.add_argument("--source", default=None, help="real | dramabox | min")
    A = ap.parse_args()
    heads = [A.head] if A.head and A.head != "both" else ["large-v2", "commercial"]
    for h in heads:
        t = RecallTable.load(h, source=A.source)
        print(f"\n=== {h} · encoder {t.encoder} · recall source '{t.source}' · "
              f"chance {t.chance:.4f} ===")
        print(f"  thresholds: strict>={t.strict_min}  hedge>={t.hedge_min}  family>={t.family_min}")
        print(f"  {'class':<20} {'strict':>7} {'family':>7}  {'tier':<8} written")
        for c in t.bursts:
            s, f = t.strict(c), t.family_recall(c)
            d = t.describe(c)
            print(f"  {c:<20} {s:7.3f} {f:7.3f}  {d['tier']:<8} ({d['written']})")
        summ = t.summary()
        print("  " + " · ".join(f"{k} {len(v)}" for k, v in summ.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
