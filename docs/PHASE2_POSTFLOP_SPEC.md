# Phase 2: Postflop evaluator extension

Implement `PostflopSubgameLeafEvaluator` as a subclass of `BaseLeafEvaluator` in `space_ko_gto.leaf_evaluator`. It must preserve the existing `(3, 1326)` reach and EV interfaces, canonical combo IDs, bounty payoff matrix, and SQLite node schema.

## Board compression and buckets

- Canonicalize suit-isomorphic flops, mapping reaches to canonical board coordinates. The target is 1,755 strategically distinct flops from 22,100 raw boards.
- Continue canonicalization incrementally through turn and river.
- Initial equity-distribution buckets: 200 on flop, 300 on turn, and 400 on river. Exact heads-up river solving is an alternative when at most 1,081 combos remain.
- Bucket quality should be measured using equity-distribution distance (EMD), not assumed from raw hand strength.

## Sizing policy by flop SPR

- `SPR <= 3.5`: 25%, 66%, jam; check-raise 2.8x or jam.
- `3.5 < SPR <= 8`: flop 33%/75%, turn 50%/100%, river 66%/jam.
- `SPR > 8`: flop 50%, turn 50%/100%, river 75%/jam; at most one raise per street.

## Multiway policy

Heads-up leaves use the full flop/turn/river tree. Three-way leaves use a reduced tree with one 50%-pot size or jam; after bet and call, the out-of-action player checks down or moves all-in according to the resolved subgame policy.

## Infrastructure and validation

The target deployment assumes at least 128 GB RAM and a high-memory GPU. Validate suit mapping invariance, bucket error, side pots, and EV conservation against exact small cases before enabling the evaluator in batch exports. Phase 1's EQR evaluator is a heuristic approximation and is not a substitute for this solver.
