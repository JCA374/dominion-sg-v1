# What a player can know, and what the teacher's sampler assumes

The handoff asked which counts and top cards the Steam observation and the public history can reconstruct, since the
rollout teacher deals the hidden cards from them.

## The network's inputs are reconstructable

`rl_features` (rl-v3 `c/rl.c`, via az-v1's `az_features`) reads my owned cards, my draw pile's contents (owned minus
hand, play and discard, which I have seen), the opponent's owned cards, the supply, coins / buys / actions, turns and
scores, plus helpers computed from those. The Steam autopilot's tracker (`autopilot/autopilot/advisor.py`) rebuilds
the same things from the screen: my cards (start + gains − trashes), the opponent's cards (start + every "Added:"
bar), my draw pile (my cards minus hand and the cards seen since the last shuffle) and the pile counts. Gap: the
autopilot does not subtract the opponent's trashes. Nothing in the features needs the opponent's zone split.

## What the simulator sampler knows beyond that

The teacher's original sampler (bv-v1 `canon` + `bv_hide`, `bv_set_obs(0)`, the "simulator" view) reshuffles the
opponent's draw pile and hand together but keeps the opponent's **discard pile exactly**, and so the exact multiset of
its draw pile + hand. A player cannot know that: the cards an opponent discards from hand at cleanup are never shown.
It also reshuffled the opponent's publicly revealed top cards (Bureaucrat's).

## The public sampler (`bv_set_obs(1)`)

- The opponent's draw pile, hand and discard pile are pooled and dealt again at their sizes (pile sizes are visible).
- The opponent's public top cards stay in place. Public top cards in the base set: the Silver a Bureaucrat gains onto
  its player's deck, and the Victory card a Bureaucrat victim puts back. Harbinger, Artisan and Sentry put back cards
  only their owner sees (tracked as my known top cards, `g_known`, as before). Tracking (`pub_observe`): a shuffle
  makes a pile unknown; growth between two decisions is private exactly when the previous decision was its owner's
  Harbinger / Artisan topdeck or Sentry choice.
- My own zones are the same in both views: I know my discard pile and my draw pile's contents; my known top cards
  stay in place, the rest of my draw pile is shuffled.

This is pessimistic: a careful player can track part of the opponent's discard pile (played cards and gains since its
last shuffle are public; the discard pile's top card is visible). The truth is between the two views; survive.py
measures how many proposed overrides depend on the difference.

## The opponent

The rollouts continue the opponent with its actual policy (c2, c1, RL-v1 or GA2), but c2's features carry no
opponent identity, so a label teaches an opponent-agnostic policy something that was measured against one specific
opponent. `bv_set_mix` continues the opponent with a fixed mixture instead (draw k uses entry k mod n); survive.py
regrades the proposed overrides with it.

Fixtures: bv-v1 `tests/test_handoff_fixes.py` (public view equivalence, private Harbinger / Artisan / Sentry tops,
public Bureaucrat tops, mixture labels reproducible) and `tests/test_replay.py` (known top cards, unknown orders).
