# Humaneness Ears Medium: annotation and procedural caption methods

This update replaces the acoustic teacher stack with a single pinned student model.
It preserves source audio, existing source annotations, Parakeet-v3 ASR, word alignment
and RMS word-end trimming. The current timed serializer adds explicit durations and
verified overlap accounting; see [PROMPT_FORMAT.md](PROMPT_FORMAT.md).

## Model and outputs

- Model: [`laion/humaneness-ears-base-medium`](https://huggingface.co/laion/humaneness-ears-base-medium).
- Revision: `818506970d93c809a3295a0a02c40dee8ff2bfc3`.
- Variant: **Medium**, based on the **Whisper Small encoder**. This product name does
  not mean OpenAI Whisper Medium. The loaded checkpoint contains **97,102,031 parameters**.
- Encoder only: transcription comes from the separate
  [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model.
- Scores: **192 regression targets**, in the model's original order and scales.
- Additional heads: characters per second, estimated burst count, 128D Orange timbre,
  250D Orange identity, 20-ms burst probabilities, event onsets/durations, and an
  event-local **53-class** burst label head.

| Raw scalar block | Count | Prefix / identifiers |
|---|---:|---|
| Emotion | 40 | `emo_*` |
| VoiceNet | 57 | `vn_*_reg` |
| Genuineness, blend, recording quality | 3 | `genuineness_0_6`, `blend_0_10`, `R_quality` |
| Additional Empathic Insight outputs | 19 | Original checkpoint target names |
| AudioBox | 4 | `audiobox:CE`, `audiobox:CU`, `audiobox:PC`, `audiobox:PQ` |
| DNSMOS | 7 | Raw and mapped SIG / BAK / OVRL; P808 MOS |
| Log burst count | 1 | `burst_count_log1p` |
| VoiceCLAP attributes | 61 | Original checkpoint target names |
| **Total** | **192** | No raw predictions clipped or discarded |

The student predicts the outputs of these original models; the teacher models are not
loaded by the new backend. See the pinned model card for training and evaluation history.
Taxonomies and source packages include [LAION voice-taxonomies](https://github.com/LAION-AI/voice-taxonomies),
[Empathic Insight Voice Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus),
[VoiceNet dimension predictors](https://huggingface.co/laion/voicenet-dimension-predictors-commercial),
[genuineness](https://huggingface.co/laion/voiceclap-commercial-genuineness),
[vocal-burst blend](https://huggingface.co/laion/voiceclap-commercial-vocalburst-blend),
[Orange timbre generation](https://github.com/LAION-AI/emotion-annotations/blob/main/generate_timbre_embeddings.py),
[AudioBox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics), and
[DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS).

## Reference sampling

The frozen normalization run uses seed **20261010** and **118,692 clips**:

| Source | Clips | Selection |
|---|---:|---|
| [S10 scaling ladder](https://huggingface.co/datasets/laion/tts-scaling-ladder-de-en) | 41,771 | Every locally available `t0` sample, 19 TAR shards |
| Gemini acting | 38,427 | Every successfully packed EN/DE production performance at the frozen snapshot; includes ten recovered ordinary-scene recordings |
| [Multilingual Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 38,494 | Seeded random shard selection and uniform eligible rows, balanced across six languages |
| **Total** | **118,692** | Equal weight per clip |

The Emilia sample contains **6,416 DE, 6,416 EN, 6,416 FR, 6,416 JA, 6,415 KO and
6,415 ZH** recordings. Up to 16 seeded random shards per language contribute quotas;
eligible rows are sampled uniformly within those shards rather than from their first rows.
Eligibility requires matching language metadata and UID prefix, duration 0.1–30 s,
and language confidence at least 0.8 where available. Exact S10 UIDs are excluded from
the Emilia sample. This is a quota-based reference, not a uniform draw from all web speech.

The manifest SHA-256 is
`3e85ac1e833e656913ea7ed66dc1f1a2dcaedf83a9eb34e72305748fb485657d`.
The local frozen manifest includes TAR byte addresses so the job reads original MP3s
directly. Existing source archives and original annotations are never rewritten. Source
paths remain in the local provenance sidecars; they are not required to interpret the
published normalization JSON.

## Measured normalization

For each of the 192 raw regression outputs and the two auxiliary scalar predictions:

```text
mean = sum(raw_i) / N
std  = sqrt(sum((raw_i - mean)^2) / N)    # population standard deviation, ddof=0
z_i  = (raw_i - mean) / std
```

Model predictions are stored in float32; summaries use float64. The preserved mixed-reference
JSON also includes medians as descriptive statistics. The pooled speech-default reference
reports means/stds only; medians cannot be pooled from source medians. **The median is not used for this z-score**.
If standard deviation is below `1e-6`, normalization is marked undefined and sidecar
z-values are null; the caption selector treats that dimension as neutral. No robust
spread floor is applied to the Medium baseline.

There are three distinct numerical representations:

| Representation | Meaning |
|---|---|
| `raw` | Original target scale, recovered using the checkpoint's training normalization |
| `training_z` | The checkpoint's standardized regression output; needed for exact model reproduction |
| `reference_z` | A fresh mean/std z-score over the frozen reference above; used for caption ranking and display |

The completed statistics file, `baseline_stats_humaneness_medium.json`, contains the
caption-compatible 40 emotion / 57 VoiceNet entries, genuineness, blend and Content
Enjoyment, plus `_raw_targets` for all 194 scalar predictions. It also reports source-
and language-specific summaries, unique audio hashes, sidecar hashes and provenance.
Exact count, unique IDs, checkpoint revision, manifest hash and finite-value coverage
are checked before normalization is written.

The reference contains training-domain material and synthetic speech. It is **not** a
held-out benchmark, a human rating study, or a representative population survey. Its
purpose is to describe how unusual a model prediction is relative to a documented mix.

## Reviewed caption design and audio evaluation

The current settings are explicit in `humaneness_caption_config.json`: **two** distinctive VoiceNet properties, up to **one** emotion at **z ≥ +0.5**, compact tags, and tempo in the general description. Sentence captions omit always-on tempo. Synonym rotation is deterministic. Demographic (`AGEV`, `GEND`, `REGS`) and genuineness wording are omitted; their raw numerical predictions remain saved. This replaces the earlier three-property/two-emotion prose configuration.

The default caption reference is **80,265 speech recordings** (41,771 S10 + 38,494 multilingual Emilia), in `baseline_stats_humaneness_medium_speech.json`. It is derived exactly from the frozen source-specific means and population variances, including between-source variance. Its provenance reports only these sources and their actual language counts, along with the filtered-manifest SHA-256 and parent hash. It is not a new independently sampled corpus. The original mixed **118,692**-clip normalizer remains in `baseline_stats_humaneness_medium.json` and can be selected as `humaneness-medium-global`. It is not silently replaced.

Pleasantness wording comes exclusively from **AudioBox Content Enjoyment**. VoiceNet ESTH stays in the numerical outputs. The blend regression describes delivery; it cannot decide whether the clip contains vocal bursts. Only selected event proposals mark bursts in the timed text. This removes the former inconsistency between a burst-free general description and nonempty event intervals.

Twenty actual audio clips (ten synthetic acting, ten Emilia) were reviewed in the earlier prose-configuration experiment with blinded candidate annotations. Ten selected the configuration; ten different clips confirmed it. The six-part 0–5 automated-reviewer composite improved from **2.675 to 3.735** on selection and **2.485 to 3.405** on confirmation. The descriptive twenty-clip mean is **2.580 versus 3.570**. Rubric weights: caption fidelity 30%, timing 25%, event precision 15%, event recall 15%, consistency 10%, readability 5%. These are automated audio-reviewer ratings, not manually adjudicated event labels or human benchmark results.

See `evaluations/procedural_caption_pilot.json` and the [English evaluation report](https://laion-moss-voice-acting-captions.static.hf.space/settings.html). The separate fifteen-combination MOSS threshold/temperature search reuses these twenty clips and has a different rubric; its scores cannot be compared directly to this procedural composite. The MOSS practical starting recommendation is greedy temperature zero with locator threshold 0.6; 0.5 favors recall and 0.7 favors precision. Threshold rankings differed between the splits, so no universal optimum was established.

## Current paired format evaluation

The [50-audio comparison](https://laion-moss-voice-acting-captions.static.hf.space/format-comparison.html) applies one shared rubric to MOSS and procedural annotations of the same recordings. A 25/25 selection/confirmation split selected compact tags before confirmation. The combined composite is **2.704/5 for both compact procedural tags and previous procedural prose**, versus **3.863/5 for uncapped greedy MOSS**. On confirmation alone, compact tags score 2.760, previous prose 2.772 and MOSS 3.646. The compact default is not a demonstrated quality improvement over prose. This rubric differs from the earlier twenty-audio study. See [the frozen metrics](evaluations/prompt-format-20261011.json) and [README](README.md) for all dimensions, uncertainty and limitations.

All 38,427 successful acting recordings pass the new serialization checks, which preserve every ASR word and numeric durations while accounting for overlap. This is a format check, not accuracy evidence. Unsupported ASR languages have no fabricated timed transcript. The [format guide](PROMPT_FORMAT.md) documents the original Humaneness Voice syntax and the explicit extension for timed overlapping events.

## Burst detection and localization

The Medium backend outputs frame probability at 20-ms resolution. Its onset/duration
heads generate raw interval proposals at onset threshold **0.5**, with
at most **32 proposals per 30-s window**. The caption default additionally selects
proposals at **0.7**, minimum duration **0.08 s**, and suppresses a lower-score
interval if its overlap exceeds half the shorter interval. Raw proposals are
retained separately from `selected_burst_event_proposals`; this is not IoU 0.5. The checkpoint does not include a `no_burst`
classification label. Its 53-way softmax describes the class conditional on a proposal;
it is not used as a separate no-burst classifier.

Every event retains `start_s`, `end_s`, onset probability and the top three class labels
with softmax probabilities. If top-1 class probability is below **0.5**, the procedural
text writes **Vocal Burst** while preserving the more specific raw predictions. This
wording threshold is a transparent heuristic, not a class-wise calibrated precision
claim. The old x2 classifier's measured recall tiers do not apply to this head.

For recordings above 30 s, windows use nominal 5-s overlap and an end-anchored final
window. Global scores are weighted using nearest-center ownership regions that partition
the actual recording duration; embeddings are averaged and L2-renormalized. Frame
probabilities are aligned to the global 20-ms grid by timestamps (including fractional
final-frame offsets), then blended using triangular context weights. Event proposals belong to
the nearest-center window at their global midpoint, preventing the same nominal event
from being included by both adjacent windows. Slight boundary disagreement can still
produce near-duplicate proposals; these are model predictions, not perfect annotations.
Long-recording burst counts use the number of retained interval proposals, while short
recordings expose the dedicated predicted count head. The original log-count scalar is
always retained separately.

## ASR and timing

Parakeet v3 is unchanged. It is used for supported known languages, including DE, EN
and FR. For **JA, KO and ZH**, it is not run and produces no synthetic word timestamps.
Existing source text, if available, is displayed in a separately labeled corpus-text
section. That text is never used to create model predictions or word boundaries.

The historical RMS end trimming keeps the original TDT end in `end_span` and adjusts
`end` only earlier, never before a word's start. Silence and timed-script verification
retain the existing implementation. Burst proposals always have both start and end.

## Reproduce annotations and statistics

Install `requirements.txt` and stage the pinned Medium snapshot and Parakeet processor /
weights. The verified environment uses PyTorch 2.9.1, Transformers 5.14.1, CUDA bfloat16,
and four CPU threads per GPU. CPU inference uses float32. Input audio is decoded,
downmixed to mono, and resampled to 16 kHz.

The listening demo also requires an FFmpeg binary for 128-kbit/s mono MP3 encoding.
Install the `imageio-ffmpeg` dependency, provide FFmpeg on `PATH`, or set
`FFMPEG_BIN=/path/to/ffmpeg` explicitly. Encoding failures stop the demo rather than
producing an audio-less publication.

```bash
# A row manifest contains id, source, source_key, language, audio_tar,
# audio_offset and audio_size. prepare_medium_reference.py builds the local reference.
python annotate_medium.py --manifest reference/annotation_manifest.jsonl \
  --out sidecars_raw --model-dir /path/to/pinned-model --rank 0 --world-size 4
# Run ranks 0–3 concurrently, each with its own CUDA-visible GPU and four CPU cores.

python compute_medium_baseline.py --sidecars sidecars_raw \
  --reference-info reference/reference_manifest_info.json \
  --output baseline_stats_humaneness_medium.json --normalized-out sidecars_normalized

python run_medium_demo.py --manifest reference/demo_manifest.json \
  --out demo_predictions --model-dir /path/to/pinned-model \
  --baseline baseline_stats_humaneness_medium.json --parakeet-dir /path/to/parakeet
python render_medium_demo.py --demo demo_predictions \
  --baseline baseline_stats_humaneness_medium.json --output docs/humaneness-medium/index.html
```

`annotate_medium.py` writes resumable **512-row Parquet segments** per rank, zstd
compression, four decode workers, batch size 32 and length-sorted windows within each
segment. Sources are read by TAR offsets, avoiding loose-file extraction and inode growth.
Normalization writes separate derived Parquets and leaves raw sidecars untouched.
Derived event end times are clamped to the stored float32 duration if rounding creates
a sub-sample overrun of at most 10 microseconds; the previous end value is retained as
`end_s_before_boundary_clamp`. Larger violations stop the job. The JSON reports the
number of these corrections. This does not change raw regression outputs or event labels.

| Sidecar columns | Contents |
|---|---|
| `sample_id`, `source`, `language`, `source_key` | Stable namespaced source identity |
| `source_reference_json`, `audio_sha256` | Original local read address and audio digest |
| `humaneness_ears_medium__raw__*` | All 192 raw scalar outputs |
| `humaneness_ears_medium__training_z__*` | All 192 training-standardized outputs |
| `humaneness_ears_medium__reference_z__*` | All 192 newly normalized outputs plus CPS and count |
| `humaneness_ears_medium__characters_per_second` | Auxiliary rate regression |
| `humaneness_ears_medium__estimated_burst_count` | Count estimate with the window convention above |
| `humaneness_ears_medium__orange_timbre_128d` | Float32 fixed-size list of length 128 |
| `humaneness_ears_medium__orange_identity_250d` | Float32 fixed-size list of length 250 |
| `humaneness_ears_medium__frame_probabilities_20ms` | Full probability sequence |
| `humaneness_ears_medium__burst_events_json` | All interval proposals and top-three labels |
| Footer metadata | Namespace, model revision, manifest hash, frame step, normalization hash |

## Listening demo and checks

The self-contained [40-clip page](https://projects.laion.ai/procedural-voice-captions/humaneness-medium/)
contains 10 DE and 10 EN seeded Gemini performances from Eric-Morris-inspired scenes,
and 20 Emilia clips: 4 DE, 4 EN, 3 FR, 3 JA, 3 KO and 3 ZH. Each audio is embedded as
mono MP3 at 128 kbit/s. Cards include actual generated captions, sentence predictions,
raw / z score tables, burst timelines and downloadable complete annotation JSON.
The fixed selection is an illustrative listening set, not a best-case ranked subset.

The GPU preflight used the actual staged checkpoint and original inference implementation:

| Check | Measured result |
|---|---|
| Public single-clip inference vs integrated backend | Maximum absolute raw-score difference **0.0** |
| Two-clip batch vs unbatched CUDA bfloat16 | Maximum difference **0.03125** training-standardized units |
| Score / event taxonomy | **192 scalar scores; 53 event classes**, finite outputs |
| Event boundaries | Start / end bounds checked |
| Parakeet integration | **59 timestamped words**, original token span ends retained and word boundaries checked |
| Normalization software smoke | **512 actual annotation rows**; all 194 nondegenerate z-columns have mean 0 / std 1; original numeric outputs and embeddings preserved |

These checks verify implementation, not emotion or event accuracy. Run
`python -m unittest -q test_medium_backend test_burst_x2 test_burst_pipeline` and
`python test_timed.py` for routing, calibration, legacy compatibility and timing checks.

## Attribution and licenses

New code, captions and prediction statistics in this update are **CC BY 4.0**, attributed
to LAION and the linked upstream model / dataset authors. Original source audio and
model weights retain their own licenses. The Emilia model card distinguishes the
CC BY-NC 4.0 Emilia release from CC BY 4.0 Emilia-YODAS; the local mixed collection does
not erase those terms. Full Gemini source archives remain private; the selected demo
performances were explicitly authorized for publication. Earlier baseline files,
model-specific experiments and source annotations are retained.
