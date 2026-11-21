# Sokoban Push Solver — 21 November 2026

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/nurdiyanto-87/sokoban-push-solver/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/nurdiyanto-87/sokoban-push-solver/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

Entry for the [make.fun competition `sokoban-solver-01-run-20261121`](https://make.fun/bounty/sokoban-solver-01-run-20261121): one program that answers every Sokoban instance of a drawn set in as few pushes as possible.

`main.ts` (Node 22, standard library only) is a room-decomposed push-optimal A* solver. Every level is a 1-wide hallway with 1–4 rooms off it; a box pushed out of its room through the door can never come back, so each room is solved on its own from its door and the room solutions are stitched together with plain player walks along the hallway. Inside a room it runs a push-based A* over (sorted box positions, canonical player cell) with an admissible heuristic (min-cost assignment of boxes to goals over single-box push distances), dead-square pruning (reverse push BFS from the goals), and corner and 2×2 frozen-block pruning. Search is budgeted by node expansions, never by the clock, so two runs produce byte-identical output.

Practice set (in the kit): **1,000,000** — push-optimal on all 5 instances — vs the house baseline plain-astar 1.0.0 at 200,000.

## Run

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output node main.ts
```

Reads `/input/<id>/level.txt` (XSB) for each instance and writes `/output/<id>/solution.txt` (moves in `u/d/l/r`).

## License

MIT — see `LICENSE`.
