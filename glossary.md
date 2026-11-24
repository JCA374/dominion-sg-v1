# Glossary

**c2** — RL-v3 run c2 at iteration 2,654: the network both branches start from, and the frozen opponent of the promotion match
**teacher** — BV-v1's rollout search: grades each candidate move by playing it to the end of the game many times
**rollout** — one copy of the game played to the end from a candidate move, with c2 choosing for both players
**draw** — one random deal of the cards a player cannot see; every candidate is played on the same draws (paired)
**incumbent** — c2's own choice at a decision
**override** — the teacher playing something other than c2's choice (only when it is better by > 2 points and > 2 SE)
**verified improvement** — a label whose winner also beats c2's choice on independent validation draws
**anchor** — a position from c2's own games where the supervised net is held to c2's policy and value
**branch A / B** — PPO from plain c2 / from the supervised c2, with the same settings, games and seeds
**manifest** — a frozen list of games (kingdom, seed; both seats) written before any game is played on it
**cohort** — labels made under identical engine, sampler, policies and grading settings; only one cohort is pooled
**bank** — 1,000 held-out positions graded again with 128–512 draws of every legal option
**public sampler** — a teacher view that deals the opponent's draw pile, hand and discard pile from public information only
**kingdom pair** — one kingdom played twice, each model in each seat; the unit of every interval here
**GA2** — the genetic-algorithm base player (portfolio-v3), a regression check
