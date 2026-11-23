# Goal 1: Review of multilingual speech datasets (MLS, VoxPopuli, etc.)

> Status: filled, 2026-11-20. Numbers for MLS and VoxPopuli come from the
> dataset cards and papers (researched Oct 2026). 
> Numbers for WorldSpeech come from the dataset card **and from our
> own metadata scan** (`scripts/scan_metadata.py`, results in
> `data/metadata/scan_stats.txt`)

## 1. Why this review

We need a speech dataset that serves three purposes at once: (a) a reliable
basis for evaluating an audio language model (ALM) in English, (b) material for
LoRA finetuning with entity pre-conditioning, and (c) text that supports named
entity recognition (NER) with spaCy — which requires cased, punctuated
transcripts. The dataset must also be large enough that strict quality filters
still leave ≥ 1,000 usable items.

Course guidance: the primary dataset should **not** be
part of Granite Speech's training data — lecturer recommended GigaSpeech,
WorldSpeech or EuroSpeech for the primary role.

## 2. Comparison

| Criterion | MLS | VoxPopuli | WorldSpeech | Common Voice | FLEURS | LibriSpeech | GigaSpeech | EuroSpeech |
|---|---|---|---|---|---|---|---|---|
| Source / paper | LibriVox audiobooks; Pratap et al., Interspeech 2020 (arXiv 2012.03411) | European Parliament plenary; Wang et al., ACL 2021 (arXiv 2101.00390) | Parliamentary proceedings (per-language sources, e.g. US House floor); DISCO group, ETH (arXiv 2605.09167, DOI 10.57967/hf/8660) | Crowd-sourced read prompts; Mozilla | Read Wikipedia sentences; Google (Conneau et al., 2022, arXiv 2205.12446) | Read audiobooks; Panayotov et al., ICASSP 2015 | Audiobooks, podcasts, YouTube; Chen et al., Interspeech 2021 (arXiv 2106.06909) | 22 European parliaments; Pfisterer et al., NeurIPS 2025 D&B (arXiv 2510.00514); official release `disco-eth/EuroSpeech` (same group as WorldSpeech) |
| Languages | 8 | 16 transcribed (+ 23 unlabelled) | 125 language–country subsets | 445 locales | 102 | 1 (EN) | 1 (EN) | 22 (European languages) |
| Hours (total / English) | ~44,660 h EN / ~1,967 h DE | 543 h EN / 282 h DE (transcribed) | `en_us`: 3,712 h (3,525.5 train + 186.6 test, own scan) | 3,843 h EN recorded (2,800 validated) / 1,502 h DE | ~12 h per language | 960 h EN | 10,000 h EN transcribed (33,005 h total) | 61k h total; 19 languages > 1k h (paper); HF mirror partial (1.49M rows, 441 GB) |
| Speech style (read, parliament, mixed) | Read (audiobooks), clean | Parliament, spontaneous, noisy | Parliament (US House floor), spontaneous | Read (crowd-recorded, quality varies) | Read, very clean | Read (audiobooks), clean | Mixed: read + spontaneous, multi-domain | Parliament, spontaneous |
| Transcripts (human or automatic) | Human, but lowercase, no punctuation | Human (`raw_text` cased + punctuated; `normalized_text` lowercase) | Both: `human_transcript` (cased, punctuated) and `asr_transcript` | Human, cased, mostly punctuated (modern validations) | Human, but lowercase and unpunctuated (`raw_transcription`, verified 2026-10-24) | Human, UPPERCASE, no punctuation | Human, but UPPERCASE; punctuation encoded as tags (`<COMMA>`, `<PERIOD>`) | Both: `human_transcript` (cased, punctuated) and `asr_transcript`, plus per-row `cer`/`wer` |
| Audio quality / metadata available | None per row | None per row | Per row: `cer`, `snr`, 4× DNSMOS, `duration`, `session_date`, source URL/timestamps | Per row: up-/downvotes, demographics (optional) | None per row | None per row | Per row: speaker id, source, category, URL — no SNR/DNSMOS | Per row: `cer`, `wer`, duration, video/transcript ids — no SNR/DNSMOS |
| License | CC BY 4.0 | CC0 | CC BY-NC 4.0 (per card; "more restrictive terms may govern") | CC0 (access via Mozilla Data Collective since Oct 2025) | CC BY 4.0 | CC BY 4.0 | HF tag `apache-2.0`, but **gated**: Terms of Access restrict use to non-commercial research/education | "unknown" on HF card; not stated |
| Suitable for ASR | Yes (huge, clean) | Yes | Yes | Yes | Yes (small) | Yes (canonical benchmark) | Yes (canonical multi-domain benchmark) | Yes |
| Suitable for TTS | Limited (text lacks prosody cues like punctuation) | Good | Good | Limited (variable quality) | Limited (lowercase text) | No (uppercase) | Limited (uppercase text, tag punctuation) | Good |
| Named entities (people, places) likely? | Low (lowercase text destroys entity cues) | High (politics: people, countries, organisations) | High (US politics: representatives, states, bills, organisations) | Medium (sentence prompts, low entity density) | Low (short Wikipedia sentences) | Low (literature, uppercase text) | Medium (entity-dense topics, but uppercase text) | High (politics: people, countries, organisations) |

## 3. Strengths and weaknesses

**MLS** — Enormous (44k+ h EN) and acoustically clean, but the transcripts are
all-lowercase with punctuation stripped. That breaks NER and any task that
relies on normal orthography. Useful as an ablation/stress corpus, not as the
main dataset.

**VoxPopuli** — Entity-dense parliamentary speech with cased, punctuated
raw_text. Weaknesses: no per-row quality metadata (quality filtering only 
possible after download), moderate EN size, and it is part of Granite Speech's 
training data, so a baseline on it is optimistic/in-domain.

**WorldSpeech** — Every row ships with quality signals and provenance, so we 
can filter before downloading audio (our scan covered all 702,054 `en_us` 
rows without downloading the ~109 GB audio payload). 
The `human_transcript` is cased and punctuated. 
Relatively new, so we train on *unknown territory*.
Weaknesses: single source per subset (`en_us` = `us_house_floor` only, 
thematically homogeneous), and no speaker IDs.

**Common Voice** — Largest speaker diversity (100k+ EN voices) and CC0, but
access moved off Hugging Face to Mozilla Data Collective in Oct 2025, quality
varies strongly between clips, and entity density in the short read prompts is
low.

**FLEURS** — Very clean read speech in 102 languages, and not in
Granite Speech's training data (good for a genuine domain shift in step 8).
But only ~12 h per language, and the transcription field is lowercase and
unpunctuated. Kept as a candidate second dataset.

**LibriSpeech** — 960 hours English ASR benchmark, but transcripts
are uppercase without punctuation and it is in Granite Speech's training data.

**GigaSpeech** — Multi-domain English (audiobooks, podcasts, YouTube; 10k h
transcribed) — strong for step-8 degradation analysis. Transcripts are uppercase
with punctuation encoded as tags (`<COMMA>`, `<PERIOD>`), so no NER use; and
there is no per-row SNR/DNSMOS, so quality filtering is only possible after
download.

**EuroSpeech** — 61k h from 22 European parliaments, built with the same 
pipeline philosophy as WorldSpeech.  
Weaknesses: only a subset of countries is published so far (no English config visible in
the Data Studio viewer); and it is *same-domain* as WorldSpeech — less useful
as a degradation contrast.

## 4. Why we chose WorldSpeech as the base

- **Filter before download:** per-row `cer`/`snr`/DNSMOS let us select clean
  items from metadata alone — decisive for building a quality-filtered package
  in limited time.
- **NER-ready text:** cased, punctuated `human_transcript` is a hard
  requirement for spaCy NER (steps 3–4) and entity pre-conditioning (steps 5–6).
- **Sufficient volume after filtering:** our scan shows ~272k train rows
  (~1,443 h) pass the candidate thresholds (CER < 0.10, SNR ≥ 12.5 dB,
  DNSMOS-ovr ≥ 3.14, 3–20 s) — far more than the ~4,000 needed; the official
  test split yields ~7,949 candidates under the stricter CER < 0.05 rule.
- **Official train/test split** supports an honest evaluation, and
  `session_date` (0 nulls) enables a date-based split.
- **For step 8 (degradation):** the homogeneous single-source domain makes a
  *different-domain* second dataset (GigaSpeech — see §5) a meaningful contrast.

Against the full field, none of the other candidates beats WorldSpeech for the
*primary* role: Common Voice, FLEURS and LibriSpeech fail the cased-text
requirement outright; GigaSpeech is uppercase with tag punctuation and has no
per-row quality metadata; EuroSpeech is the closest competitor (same pipeline
philosophy, cased transcripts). WorldSpeech is the only one (compared with 
GigaSpeech and EuroSpeech) with cased text and per-row quality metadata. 
VoxPopuli, LibriSpeech, Common Voice and MLS are disqualified by the rule;
all four appear in Granite Speech training-data table (CommonVoice,
MLS, LibriSpeech/LibriHeavy, VoxPopuli).

## 5. Second dataset (for step 8)

**For now: GigaSpeech** (English, 10,000 h transcribed from audiobooks,
podcasts and YouTube) — a different domain from US parliamentary
speech, so degradation under domain shift is measurable.

## References

- WorldSpeech dataset card: https://huggingface.co/datasets/disco-eth/WorldSpeech (license, schema, arXiv 2605.09167, DOI 10.57967/hf/8660)
- MLS: Pratap et al., *MLS: A Large-Scale Multilingual Dataset for Speech Research*, Interspeech 2020, arXiv 2012.03411; https://www.openslr.org/94
- VoxPopuli: Wang et al., *VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation*, ACL 2021, arXiv 2101.00390
- Common Voice: https://commonvoice.mozilla.org (statistics via the live Common Voice API, accessed Oct 2026)
- FLEURS: Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, 2022, arXiv 2205.12446; https://huggingface.co/datasets/google/fleurs
- LibriSpeech: Panayotov et al., *LibriSpeech: An ASR corpus based on public domain audio books*, ICASSP 2015; https://www.openslr.org/12
- GigaSpeech: Chen et al., *GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio*, Interspeech 2021, arXiv 2106.06909; card https://huggingface.co/datasets/speechcolab/gigaspeech (gated, Terms of Access non-commercial); GitHub https://github.com/SpeechColab/GigaSpeech
- EuroSpeech: Pfisterer et al., *EuroSpeech: A Multilingual Speech Corpus*, NeurIPS 2025 Datasets & Benchmarks, arXiv 2510.00514; HF mirror https://huggingface.co/datasets/sam8000/EuroSpeech (license "unknown")
- Own measurements: `scripts/scan_metadata.py`, `data/metadata/scan_stats.txt` (scan of 2026-11-20)
- Granite Speech model card (training-data table): https://huggingface.co/ibm-granite/granite-speech-5.0-2b (released 2026-10-20; training corpora verified 2026-11-20 — this is also the model used in the course labs `mmai_07`/`mmai_08`)
