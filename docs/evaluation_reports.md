# Qwen3-ASR Evaluation with Named-Entity Context

## Executive summary

Across the same 500 samples, named-entity context reduced corpus WER from
0.127155 to 0.122084 (3.99% relative) and corpus CER from 0.084403 to 0.080613
(4.49% relative). The gain is real but modest: WER was unchanged on 425 samples,
improved on 53, and worsened on 22. An approximate entity-mention diagnostic
also improved, although the CSV does not support a reliable official entity
accuracy rate.

## Metrics

| Metric | Without context | With entity context | Change |
|---|---:|---:|---:|
| Samples | 500 | 500 | — |
| Corpus WER | 0.1271550 (12.7155%) | 0.1220844 (12.2084%) | −0.0050706 (−0.5071 pp; −3.99% relative) |
| Corpus CER | 0.0844030 (8.4403%) | 0.0806127 (8.0613%) | −0.0037904 (−0.3790 pp; −4.49% relative) |
| Mean per-sample WER | 0.1779720 (17.7972%) | 0.1732002 (17.3200%) | −0.0047718 |
| Mean per-sample CER | 0.1448401 (14.4840%) | 0.1427431 (14.2743%) | −0.0020969 |
| Median per-sample WER | 0.0833333 | 0.0833333 | No change |
| Median per-sample CER | 0.0417757 | 0.0415587 | −0.0002170 |
| Samples with perfect WER | 114 (22.8%) | 118 (23.6%) | +4 |
| Inference time | Not recorded | Not recorded | — |

Alignment hits increased from 11,932 to 12,000. Substitutions fell from 580 to
508, while deletions rose from 307 to 311 and insertions from 743 to 746. At
the sample level, WER improved on 53/500 samples, was unchanged on 425/500, and
worsened on 22/500. For CER, the counts were 58/419/23.

As an additional diagnostic, I split the `entities` field on commas and looked
for each fragment in the predictions, ignoring case and punctuation. The
fragment hit rate was 695/870 without context (79.89%) and 752/870 with context
(86.44%), a 6.55 percentage-point increase. This is not an official entity
recognition rate: the CSV serializes the entity list as comma-separated text,
so names containing commas may be counted as multiple mentions. The metrics
summary does not define or record an entity metric.

## Behavior and observed cases

The following examples are normalized as they appear in `predictions.csv`:

- **Corrected proper names:** In `us_en_20170105_1416`, the reference is
  “phyllis schneck dorwan stoddard … gabriel gabe zimmerman”. Without context,
  Qwen transcribed “phyllis schnack dorwin stoddard”; with context, it correctly
  transcribed “phyllis schneck dorwan stoddard”. WER fell from 0.20 to 0.00 and
  CER from 0.025 to 0.00.
- **Rare surname:** In `us_en_20201117_0884`, “la la mafa” became “la lamalfa”,
  and an extra word at the start disappeared. WER fell from 0.5556 to 0.2222
  and CER from 0.1170 to 0.0638. Context helped, but did not eliminate all
  errors.
- **Context degrades an almost-correct transcription:** In
  `us_en_20220713_2240`, the no-context output preserves nearly all of the
  reference (“… and 382 printed in part a of house report number 117405 …”),
  with WER 0.0476. With context, the output drops “and 382 printed in part a
  of house report” and retains “house 117 smith washington”; WER rises to
  0.6190.
- **Repeated insertion with names and places:** In `us_en_20230330_0647`, the
  baseline exactly matches the reference. With context, the model prepends
  “last summer 53 texas” and then transcribes the sentence again, raising WER
  from 0.00 to 0.2105. This is an insertion relative to the reference; the
  audio has not been re-listened to verify what was said.
- **An entity can improve without rescuing the sample:** In
  `us_en_20180314_0345`, context causes “5 minutes” to appear at the beginning,
  but both outputs also contain a long transcription that does not match the
  short reference. WER, already extremely high, rises from 11.6667 to 12.3333.
  This illustrates the sensitivity of per-sample WER on very short segments
  and that correctly transcribing an entity does not guarantee a correct
  transcript.

## Technical conclusions and next steps

- The results support a modest aggregate benefit, driven mainly by fewer
  substitutions. They do not support assuming a uniform improvement or using
  context without regression checks.
- This evaluation uses **entities extracted by spaCy from the reference
  transcript** to build the prompt. This is oracle context, not a deployable
  flow with entities detected from audio or an independent source. These
  figures therefore do not measure end-to-end NER+ASR performance and may
  overstate the benefit available in production.
- For fine-tuning, retain the paired comparison and add breakdowns by entity
  type (person, place, organization, and number), frequency, and segment
  length. Measure entity accuracy using a structured representation (for
  example, a JSON list rather than comma-separated text) and report exact
  mentions corrected, preserved, and degraded.
- Compare oracle context with context from a real NER pipeline, and include
  both a no-context baseline and a post-fine-tuning ablation. Record latency by
  condition and manually review samples with strong insertions or regressions
  before attributing them to model hallucinations.

## Artifacts reviewed

- `results/step2_baseline/baseline_en_qwen3_asr/metrics_summary.json`
- `results/step2_baseline/baseline_en_qwen3_asr/predictions.csv`

The run uses `Qwen/Qwen3-ASR-1.7B-hf` on the `test` split of
`disco-eth/WorldSpeech`, subset `en_us`.
