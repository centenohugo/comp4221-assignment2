# COMP4221 Programming Assignment Report

- **Name:** Hugo Centeno Sanz
- **Student ID:** 21431159
- **UST email:** hcentenosanz@connect.ust.hk
- **AI assistance disclosure:** Used Opus 5.5 for coding explanation before implementation. Run all execution commands myself to use it as checkpoints of knowing what I was doing.

  Used Opus 5.5 for report structure help.

- https://github.com/centenohugo/comp4221-assignment2

## Part A: Normalize and validate FAQ records

### Counts (`data/counts.json`)

| Stage | Records |
| --- | ---: |
| Raw | 120 |
| Valid before deduplication | 110 |
| Unique | 100 |

### Normalization examples

| ID | Field | Before | After |
| --- | --- | --- | --- |
| F007 | question | `"  What  is the booking limit for study room S07&#63;\t"` | `"What is the booking limit for study room S07?"` |
| F013 | question | `"  Ｗｈａｔ  ｉｓ ｔｈｅ ｂｏｏｋｉｎｇ ｌｉｍｉｔ ｆｏｒ ｓｔｕｄｙ ｒｏｏｍ Ｓ13?\t"` | `"What is the booking limit for study room S13?"` |


### Conflicting answers

**Risk.** Deduplicated key ignores the answer. If two records had the same ID and
question but different answers (e.g. `2 hours` vs. `3 hours`), the first one would
be kept and the other lost.

**Detection (not implemented).** Group the valid records by the deduplicate key and
collect the normalized answers. Any group with more than one
distinct answer should be manually reviewed (or by an LLM). 

## Part B: Nucleus probability filtering

### Worked example

Logits `ln([0.50, 0.30, 0.15, 0.05])`, temperature 1, so the softmax gives back
`[0.50, 0.30, 0.15, 0.05]`. Already sorted, the cumulative sums are
`[0.50, 0.80, 0.95, 1.00]`.

| `top_p` | Retained indices | Kept mass | Output (3 d.p.) |
| --- | --- | ---: | --- |
| 0.70 | 0, 1 | 0.80 | `[0.625, 0.375, 0.000, 0.000]` |
| 0.90 | 0, 1, 2 | 0.95 | `[0.526, 0.316, 0.158, 0.000]` |

The output is the kept probabilities divided by the kept mass, e.g. 0.50 / 0.80 = 0.625.

### Crossing token and variable support

The crossing token is the one whose addition makes the cumulative sum reach or pass
`top_p`. For `top_p=0.70`, token 0 alone gives 0.50, which is not enough; adding
token 1 gives 0.80 ≥ 0.70, so token 1 is the crossing token and is kept. Without it
the kept mass would stay below `top_p`, which is what we want to avoid. If a prefix
sums to exactly `top_p`, it already reaches the threshold and no further token is added.

The number of kept tokens is not fixed because it depends on how the probability
is spread.

### Checker

MISSING INCLUDE SCREENSHOT

## Part C: Final response supervision and smoothed loss

### C1: Supervise the last original response

Example with 10 tokens, two assistant spans and `pad_token_id = 0`:

- `input_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- `assistant_spans = [(2, 4), (7, 10)]`, so the last response covers positions 7, 8, 9.

**Without truncation (`max_length = 12`).** The sequence is shorter than 12, so 2
padding positions are added at the end.

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `input_ids` | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 0 | 0 |
| `labels` | -100 | -100 | -100 | -100 | -100 | -100 | -100 | 17 | 18 | 19 | -100 | -100 |
| `attention_mask` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |

Only the last span (positions 7–9) is supervised. The earlier response (2–3) is
`-100`. Padding has attention 0 and label `-100`.

**With truncation that removes the final response (`max_length = 6`).** Positions
6–9 are cut, so no padding is needed.

| Position | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| `input_ids` | 10 | 11 | 12 | 13 | 14 | 15 |
| `labels` | -100 | -100 | -100 | -100 | -100 | -100 |
| `attention_mask` | 1 | 1 | 1 | 1 | 1 | 1 |

The trainer skips this example because it has no valid targets.

### Context vs. supervised targets

- **Context** (`input_ids` with attention 1) is what the model *reads*: system and
  user messages, role headers and earlier assistant turns. It helps the model know
  what to answer.
- **Supervised targets** (`labels` different from `-100`) are what the model *learns
  to predict*: only the final response, including its EOS. The loss ignores every
  other position.

### C1 checker

# Missing screenshot!

### C2: Conversation-balanced label smoothing

TODO

## Part D: Fine-tuning and grounded-format evaluation

TODO
