# Qwen3-ASR Evaluation with Named-Entity Context

## Executive summary

On 1,000 WorldSpeech (`en_us`) samples, adding spaCy-extracted entities reduced corpus WER from 12.70% to 12.32% (−0.379 percentage points; −2.99% relative) and corpus CER from 8.12% to 7.85% (−0.272 pp; −3.35% relative). The perfect-transcription rate increased from 22.2% to 23.5%. This is a small but consistent aggregate gain, not a universal improvement: WER improved on 88 samples, was unchanged on 867, and worsened on 45. Because the setup uses entities from the reference transcript, it measures oracle context rather than a deployable NER+ASR system.

## Experimental setup

- **Model:** `Qwen/Qwen3-ASR-1.7B-hf`, English (`en`), with up to 256 new tokens; paired comparison with no prompt and with an entity prompt.
- **Dataset:** `disco-eth/WorldSpeech`, `en_us` subset, `test` split. `metrics_summary.json` declares `samples: 1000`, and the CSV contains 1,000 prediction rows.
- **Selection and filters:** up to 1,000 samples, seed 42, and the frozen manifest `data/sample_ids.json`; non-empty transcript, `dnsmos_ovr >= 3.0`, and duration from 3 through 15 seconds inclusive. Audio is converted to mono and resampled to 16 kHz.
- **NER and context:** `src/model.py` loads `en_core_web_trf` and falls back to `en_core_web_sm` if unavailable. It extracts unique entities in source order from the reference transcript and passes them as the second inference's prompt. The saved results do not identify which spaCy model was installed; context comes from the reference text, not the audio.
- **Evaluation:** lowercase normalization, punctuation removal, and whitespace collapsing (`lab6`); per-sample WER/CER and corpus metrics. Error counts are based on word alignment.

## Quantitative results (No context vs. NER context)

| Metric | No context | NER context | Change |
|---|---:|---:|---:|
| Samples | 1,000 | 1,000 | — |
| Corpus WER | 0.127018 (12.7018%) | 0.123226 (12.3226%) | −0.003792 (−0.3792 pp; −2.99% relative) |
| Mean WER | 0.169639 (16.9639%) | 0.165780 (16.5780%) | −0.003859 |
| Median WER | 0.089087 (8.9087%) | 0.084524 (8.4524%) | −0.004563 |
| Corpus CER | 0.081234 (8.1234%) | 0.078509 (7.8509%) | −0.002725 (−0.2725 pp; −3.35% relative) |
| Mean CER | 0.126429 (12.6429%) | 0.124369 (12.4369%) | −0.002060 |
| Median CER | 0.043597 (4.3597%) | 0.041921 (4.1921%) | −0.001676 |
| `perfect_wer_ratio` | 0.222 (222; 22.2%) | 0.235 (235; 23.5%) | +0.013 (+13 samples; +1.3 pp) |
| Substitutions | 1,225 | 1,117 | −108 |
| Deletions | 653 | 651 | −2 |
| Insertions | 1,371 | 1,384 | +13 |
| Hits (`hits`) | 23,701 | 23,811 | +110 |

The number of aligned reference words (`N`) is 25,579 in both conditions. Overall, word errors (substitutions + deletions + insertions) decreased by 97, mainly because of fewer substitutions; insertions increased slightly. In the paired CSV analysis, WER improved in 88/1,000 cases (8.8%), worsened in 45 (4.5%), and was unchanged in 867 (86.7%). For CER, it improved in 94, worsened in 44, and was unchanged in 862.

## Error analysis and examples

The following text and error values are normalized as they appear in `predictions.csv`.

**Entities corrected or helped by context**

- `us_en_20201117_0884` — **Reference:** “missouri representative huffman representative thompson representative lamalfa from california”. **No-context prediction:** “of missouri representative huffman representative thompson representative la la mafa of california”. **Context prediction:** “missouri representative huffman representative thompson representative la lamalfa of california”. **Entities:** `Missouri, Representative Huffman, LaMalfa, California`. WER falls from 0.5556 to 0.2222: the surname is recovered, though other errors remain.
- `us_en_20171010_0356` — **Reference:** “of student loans out there nearly 900 billion of that 13 trillion”. **No-context prediction:** “of student loans out there nearly nine hundred billion of that one point three trillion”. **Context prediction:** “of student loans out there nearly 900 billion of that 13 trillion”. **Entities:** `Nearly $900 billion, $1.3 trillion`. WER falls from 0.4167 to 0.0.
- `us_en_20250714_0461` — **Reference:** “s 1596 to rename the anahuac national wildlife refuge located in the state of texas as the jocelyn nungaray national wildlife refuge”. **No-context prediction:** “senate 1596 an act to rename the anahuac national wildlife refuge located in the state of texas as the jocelyn nungarrete national wildlife refuge”. **Context prediction:** “senate one thousand five hundred and ninety six an act to rename the anahuac national wildlife refuge located in the state of texas as the jocelyn nungaray national wildlife refuge”. **Entities:** `1596, the Anahuac National Wildlife Refuge, the State of Texas, the ``Jocelyn Nungaray National Wildlife Refuge''`. The surname is corrected, but expanding the number and adding words raises overall WER from 0.1818 to 0.4545.

**Persistent phonetic errors and regressions**

- `us_en_20170428_0404` — **Reference:** “ago washington got weaker they were able to get ahold of the doctor who was one of george washingtons best friends in the world dr craik”. **No-context prediction:** “and washington got weaker well they were able to get a hold of a doctor that was one of george washingtons best friends in the world doctor craik craik”. **Context prediction:** “and washington got weaker well they were able to get a hold of a doctor that was one of george washingtons best friends in the world doctor craik c r a i k”. **Entities:** `Washington, one, George Washington's, Craik`. The phonetic spelling of the surname still does not match the reference; WER rises from 0.3077 to 0.4615.
- `us_en_20220713_2240` — **Reference:** “377 378 379 380 381 and 382 printed in part a of house report 117405 offered by mr smith of washington”. **No-context prediction:** “377 378 379 380 381 and 382 printed in part a of house report number 117405 offered by mr smith of washington”. **Context prediction:** “377 378 379 380 381 house 117 smith washington”. **Entities:** `377, 378, 379, 380, 381, 382, House, 117, Smith, Washington`. Context removes much of an otherwise nearly correct sentence: WER rises from 0.0476 to 0.6190.

**Outliers**

- `us_en_20180314_0345` has the short reference “for 5 minutes”. Without context, the model produces a long response beginning “well thank you mr speaker…” (WER 11.6667); with the `5 minutes` context, it prepends “for five minutes” but retains the long remainder (WER 12.3333). The short reference makes the WER extreme when the output is misaligned; context does not resolve the mismatch.
- `us_en_20211209_0522` also has WER 10.5 in both conditions: the reference is “the recognition”, while the much longer transcription begins “seek recognition designated by miss underwood…”. These are mismatches against the reference; without listening to the audio, they cannot be confidently attributed to model hallucination.

## Technical conclusions and next steps

Entity context modestly improves aggregate metrics and exact transcription, primarily by reducing substitutions of names and numbers. However, it can also cause substantial regressions and does not prevent phonetic errors, insertions, or outputs disproportionate to short references. It should not be treated as a guaranteed per-sample improvement.

For the entity-preconditioned fine-tuning phase:

1. Retain this paired evaluation as the baseline and evaluate the fine-tuned model using the same IDs, filters, and normalization.
2. Isolate the effect of fine-tuning from prompting with no-context and context ablations; compare oracle reference entities with entities from a source available at inference time.
3. Report results by entity type and frequency, duration, and reference length; measure corrected, preserved, and degraded mentions using structured entities rather than comma-delimited text.
4. Manually inspect large regressions and short references; record latency and the spaCy model used.

## Artifacts analyzed

- `results/step2_baseline/baseline_en_qwen3_asr/metrics_summary.json` — `samples: 1000` metadata and aggregate metrics.
- `results/step2_baseline/baseline_en_qwen3_asr/predictions.csv` — 1,000 paired predictions, entities, references, and per-sample WER/CER.
- `configs/eval_config.yaml`, `src/dataset.py`, `src/model.py`, and `src/evaluate.py` — pipeline configuration and behavior.

No separate entity or error files were found; entities and examples were analyzed from the CSV columns.
