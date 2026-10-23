# COMP4221 Programming Assignment Report

- **Name:** Hugo Centeno Sanz
- **Student ID:** 271597004
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

TODO

## Part C: Final response supervision and smoothed loss

TODO

## Part D: Fine-tuning and grounded-format evaluation

TODO
