# OptiStop

**AI-assisted optimal stopping prototype using fuzzy logic, regime detection, and game-theoretic decision making.**

## Problem

When opportunities arrive sequentially, a decision-maker must choose between **STOP** and **CONTINUE** without knowing what will appear next. A fixed threshold can be too rigid when opportunity quality changes over time.

## Approach

OptiStop combines:

1. **Optimal stopping** — frames the decision as STOP vs CONTINUE.
2. **Fuzzy opportunity scoring** — combines multiple noisy attributes into a gradual opportunity-quality score.
3. **Regime detection** — estimates whether the opportunity environment is improving, stable, or deteriorating.
4. **Game-theoretic payoff framing** — accounts for search cost, competition pressure, risk aversion, and the payoff of continuing the search.
5. **Adaptive warm-up / exploration** — adjusts early exploration using observed evidence.

## Prototype result

The current controlled benchmark uses five synthetic regimes:

- IID
- Late Surge
- Early Surge
- Increasing Trend
- Decreasing Trend

Using 10,000 trials per scenario, 10 opportunities per sequence, and a fixed seed of 42, the current Game-Fuzzy v7 benchmark reported:

- **Average score:** 79.42
- **Average gain vs Classical:** +12.83 points
- **Scenario wins:** 4/5
- **Relative gain vs Classical:** +19.3%

The method does **not** win every scenario: in IID, Classical remains ahead. This is presented as controlled simulation evidence, not proof of universal optimality.

## Running the prototype

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```text
OptiStop/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── docs/
│   └── architecture.md
└── benchmark/
    └── README.md
```

## Hackathon framing

**One-line pitch:** OptiStop makes the decision to stop searching adaptive by combining opportunity quality, changing regimes, uncertainty, and strategic search costs.

## Limitations

This is a hackathon/research prototype. The benchmark scenarios are synthetic and the game-theoretic component is a payoff framing rather than a claim of a universal Nash equilibrium. Real-world validation would require domain-specific data, calibration, and robustness testing.
