# sg-v1 — search-guided training of RL-v3 c2

**🌐 Live site: https://jca374.github.io/dominion-sg-v1/**

Does teaching RL-v3 c2 the verified improvements of the BV-v1 rollout teacher, and then continuing PPO, give a
stronger model than PPO alone from the same start? This trainer carries out the handoff
`DOMINION_C2_SEARCH_TRAINING_HANDOFF.md` (workspace root) and its goal brief `GOAL_C2_SEARCH_TRAINING.md`.

- **Branch A (control):** RL-v3 c2 iteration 2654 → league PPO.
- **Branch B (search-guided):** c2 2654 → supervised pairwise update from verified teacher labels → the same PPO
  schedule and game budget as A.
- **Promotion gate:** a fresh 2,000-kingdom both-seat match against frozen c2, ≥ 53% with the paired 95% lower bound
  above 50%, no regression against GA2 and none on the card slices.

## Layout

The trainers are pinned submodules; nothing of theirs is copied here.

| Path | What |
|---|---|
| `engine/` | the rules (dominion-engine), the same commit as in both trainers |
| `rl-v3/` | the net, PPO (`train.py`), its frozen opponents |
| `bv-v1/` | the rollout teacher (`c/bv.c`), label collection, matches, the validation bank |
| `manifests/` | frozen game manifests (ablation, dev, matrix, promotion) and the shared kingdom exclusion list |
| `ablation.py`, `ablation_report.py` | step 1: the matched teacher ablation |
| `anchors.py`, `supervise.py` | step 3: c2's own positions as anchors; the pairwise update toward verified gains |
| `ppo.py` | step 4: matched PPO, A and B, two training seeds |
| `netmatch.py`, `bank_eval.py`, `survive.py` | matches between nets, bank scores, the observation / opponent check |
| `results/` | every measurement (JSON); `models/` released and candidate nets |

Python is rl-v3's venv (`../rl-v3/.venv/bin/python`, or set `SG_PY`). Long jobs run as `systemd-run --user` units
with logs in `logs/` and resume flags (`--resume` / `--continue`); see `RESUME.md`.

## Fixes made before the long runs

All of the handoff's "Fixes before a long run" landed in the trainers, each with a test (rl-v3
`tests/test_ppo_fixes.py`, bv-v1 `tests/test_handoff_fixes.py` and `tests/test_bank.py`):

- **PPO:** the update divides the logits by the sampler's temperature (first ratio error at temp 0.7: 0.048 before,
  3e-7 after); errored and record-overflowed games give no training records; the stream seed no longer depends on the
  run name, worker results come back in order, and minibatch shuffles are seeded, so two runs with one seed write
  identical nets; frozen c2 and c1 can join the league; a shared kingdom exclusion list; `--init_from` needs an
  explicit `--stage` and checks at step zero that the weights equal the source and the first PPO ratio is 1; game
  counts, optimizer updates and CPU hours are logged.
- **Teacher:** buys and actions share one grader; a true action label-only mode (c2 keeps control; full rows with
  kind, ordinal, features, legal and graded masks, grades and paired SE per candidate, draws, caps); one failure
  policy (an engine error in any rollout: no label, no override); zero-SE ties draw on; independent validation draws
  for the graded winner; a public sampler (the opponent's draw pile, hand and discard pile pooled; Bureaucrat's public
  top cards kept) and an opponent mixture for the rollouts, with fixtures (Sentry / Harbinger / Artisan private,
  Bureaucrat public, unknown orders).
- **Data:** label shards carry a cohort id (engine and sources, sampler, policies, grading); the bank requires replay
  identity (features, kind, incumbent, legal mask) and fails on mismatches, scores at the deployment margin and counts
  strict shortlist misses by value; a match writes its manifest (source and library hashes, the game list's hash)
  before game one, checks resumed rows and scores completed seat pairs only.

## Results

Every number below: n games, paired 95% interval (kingdom pairs, both seats), development numbers marked.
Live table; `results/` has the files.

| Step | Measurement | Result |
|---|---|---|
| 1 | Ablation: plain c2 mirror vs c2 (600 games) | 50.0% [50.0, 50.0] |
| 1 | Ablation: buy teacher on c2 vs c2 (600 games; 3.9% of buys overridden; latency p50 0.78 s, p95 2.56 s) | **56.0%** [53.0, 59.0] |
| 1 | Ablation: action teacher on c2 vs c2 (600 games; 1.9% of actions overridden; p50 0.40 s, p95 1.13 s) | **55.1%** [52.4, 57.9] |
| 1 | Ablation: both on c2 vs c2 (600 games; p50 0.47 s, p95 1.86 s) | **58.8%** [55.6, 62.1] |
| 1 | Paired: both − buy / both − action / action − buy (300 kingdom pairs) | +2.8 [−1.2, +6.7] / +3.8 [−0.4, +7.8] / −0.9 [−4.9, +3.2] points |
| 2 | Pilot labels: buy-only (the simplest arm that reproduces a gain), 20,000 positions, 32 validation draws | collecting |

## Optimizations measured

| Change | Result |
|---|---|
| One hidden-state preparation per draw (bv-v1 `bv_set_prep`) | identical grades, moves, caps and errors; preparations 6.6× fewer, wall time unchanged (rollouts dominate, ~1.6 ms each) |
| No record buffers in score-only games (rl-v3 eval) | 3–5% faster per evaluation game |
| GA genome resolution for NET seats | 0.04 ms per call: not worth changing |
| PPO record concatenation releasing each part | one copy of the iteration's records at the peak instead of two (same data) |
