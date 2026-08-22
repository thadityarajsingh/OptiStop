# OptiStop Architecture

## Decision pipeline

```text
Sequential opportunities
        |
        v
Opportunity attributes
        |
        v
Fuzzy opportunity scoring
        |
        v
Regime belief
Improving / Stable / Deteriorating
        |
        v
Strategic payoff layer
STOP vs CONTINUE
        |
        v
Final decision
```

## Main concepts

### Optimal stopping
The decision-maker observes opportunities one at a time and must decide whether to accept the current opportunity or continue searching.

### Fuzzy logic
Multiple uncertain attributes are converted into gradual scores rather than relying on one hard threshold.

### Regime detection
Observed trend information is used to distinguish improving, stable, and deteriorating conditions.

### Game-theoretic decision
The prototype compares the strategic value of stopping with continuing, including search cost, competition pressure, risk aversion, and uncertainty.

### Validation
The benchmark compares Game-Fuzzy v7 with Classical, Adaptive, Regime-Aware, and an Oracle upper-bound reference across controlled synthetic scenarios.
