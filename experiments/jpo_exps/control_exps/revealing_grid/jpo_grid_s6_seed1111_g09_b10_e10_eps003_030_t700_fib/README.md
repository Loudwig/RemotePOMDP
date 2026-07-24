# JPO gamma=0.9 full grid with FIB initial upper bound

This experiment reproduces the 100 points of
`jpo_grid_s6_seed1111_g09_b10_e10_eps003_030_t700` with exactly the same MDP,
beta/epsilon grid, SARSOP precision, and 700-second solver cap. The only
algorithmic change is `initial_upper_bound = "fib"`.

The first scheduled points are `g0_b09_e00`, `g0_b06_e00`, `g0_b09_e09`, and
`g0_b06_e09` so that the high-beta comparison becomes available early. After
those points, the remaining grid is processed from high beta to low beta.

Run or resume with:

```bash
python3 -B run_experiment.py --workers 2
```

While the solver is running, compare its latest flushed history against the
fully-observable-upper experiment with:

```bash
python3 -B compare_live.py
```
