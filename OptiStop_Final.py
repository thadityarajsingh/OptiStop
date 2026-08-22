# Final prototype validation scenarios
SCENARIOS = [
    "IID",
    "Late Surge",
    "Early Surge",
    "Increasing Trend",
    "Decreasing Trend",
]

import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OptiStop Final Dashboard", page_icon="🛑", layout="wide")

# ============================================================
# Fuzzy scoring
# ============================================================
def trap(x, a, b, c, d):
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / max(b - a, 1e-9)
    return (d - x) / max(d - c, 1e-9)


def tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    return (x - a) / max(b - a, 1e-9) if x < b else (c - x) / max(c - b, 1e-9)


def fuzzy_score(v, fit, growth, org, conv):
    hv, hf, hg, ho, hc = [trap(x, 55, 70, 100, 100)
                           for x in (v, fit, growth, org, conv)]
    mf, mg = tri(fit, 25, 50, 75), tri(growth, 25, 50, 75)
    rules = [
        (min(hv, hf, hg, ho), 95),
        (min(hv, hf, ho), 88),
        (min(hf, hg), 82),
        (min(hv, hg), 80),
        (min(ho, hf), 78),
        (min(mf, hv), 68),
        (min(mg, hf), 65),
    ]
    den = sum(m for m, _ in rules)
    return sum(m * s for m, s in rules) / den if den else np.mean(
        [v, fit, growth, org, conv]
    )


# ============================================================
# Data generation
# ============================================================
def generate_scenarios(trials, n, seed):
    """Generate the five fixed validation scenarios with deterministic seeds.

    Returns a dictionary so the final dashboard can run one scenario at a time
    while keeping the same trial count, sequence length, and base seed.
    """
    return {
        scenario: generate(trials, n, scenario, seed + i)
        for i, scenario in enumerate(SCENARIOS)
    }


def generate(trials, n, scenario, seed):
    r = np.random.default_rng(seed)
    if scenario == "IID":
        return r.normal(70, 12, (trials, n)).clip(0, 100)
    if scenario == "Late Surge":
        k = max(1, int(0.6 * n))
        return np.c_[
            r.normal(60, 6, (trials, k)),
            r.normal(86, 7, (trials, n - k)),
        ].clip(0, 100)
    if scenario == "Early Surge":
        k = max(1, int(0.4 * n))
        return np.c_[
            r.normal(86, 7, (trials, k)),
            r.normal(60, 6, (trials, n - k)),
        ].clip(0, 100)

    trend = (
        np.linspace(45, 90, n)
        if scenario == "Increasing Trend"
        else np.linspace(90, 45, n)
    )
    return (trend + r.normal(0, 5, (trials, n))).clip(0, 100)


# ============================================================
# Baselines
# ============================================================
def classical(seq):
    n = len(seq)
    k = max(1, int(n / math.e))
    bench = max(seq[:k])
    for i in range(k, n):
        if seq[i] > bench:
            return float(seq[i]), i
    return float(seq[-1]), n - 1


def adaptive(seq, frac=0.4, margin=1):
    n = len(seq)
    k = max(2, int(n * frac))
    obs = list(seq[:k])

    for i in range(k, n):
        cur = float(seq[i])
        rem = n - i - 1
        if rem == 0:
            return cur, i

        q = 1 - 1 / max(rem + 1, 2)
        predicted = float(np.quantile(obs, q))
        threshold = max(max(obs), predicted + margin)

        if cur >= threshold:
            return cur, i

        obs.append(cur)

    return float(seq[-1]), n - 1


def classical_secretary(seq):
    """Compatibility wrapper used by the final judge-facing dashboard."""
    return classical(seq)


def adaptive_baseline(seq, explore_frac=0.40, safety_margin=1.0):
    """Compatibility wrapper used by the final judge-facing dashboard."""
    return adaptive(seq, frac=explore_frac, margin=safety_margin)


# ============================================================
# Regime detector + belief state
# ============================================================
def slope(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=float)
    return float(np.polyfit(t, x, 1)[0])


def volatility(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return max(float(np.std(x)), 5.0)
    return max(float(np.std(np.diff(x))), 1.0)


def r2_trend(x):
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x) < 1e-9:
        return 0.0
    t = np.arange(len(x), dtype=float)
    coef = np.polyfit(t, x, 1)
    pred = np.polyval(coef, t)
    denom = np.sum((x - x.mean()) ** 2)
    if denom <= 1e-9:
        return 0.0
    return float(1 - np.sum((x - pred) ** 2) / denom)


def regime_belief(observed, slope_threshold=3.0):
    """
    Soft belief over latent Improving / Stable / Deteriorating states.
    Only the observed prefix is used.
    """
    x = np.asarray(observed, float)
    s = slope(x)
    vol = volatility(x)
    scale = max(float(slope_threshold), vol / 2.0, 1.0)

    logits = np.array([s / scale, 0.0, -s / scale], dtype=float)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()

    labels = np.array(["Improving", "Stable", "Deteriorating"])
    return dict(zip(labels, p)), s, vol, r2_trend(x)


def regime_confidence(observed, slope_threshold=3.0):
    """
    Confidence that the observed prefix supports a directional regime.

    Belief concentration alone can be misleading on noisy IID sequences.
    We therefore combine:
      - belief concentration (max regime probability), and
      - linear trend quality (R²).

    A flat/noisy sequence should remain low-confidence even if a random
    slope slightly favors Improving or Deteriorating.
    """
    beliefs, _, _, r2 = regime_belief(observed, slope_threshold)
    concentration = float(max(beliefs.values()))
    trend_quality = float(np.clip(r2, 0.0, 1.0))

    # Keep confidence conservative unless both classification and trend
    # consistency support the regime.
    confidence = concentration * (0.5 + 0.5 * trend_quality)
    return float(np.clip(confidence, 0.0, 1.0))


def classify_regime(observed, slope_threshold=3.0):
    beliefs, s, _, _ = regime_belief(observed, slope_threshold)
    return max(beliefs, key=beliefs.get), s


# ============================================================
# v3 regime-aware baseline
# ============================================================
def regime_aware(seq, explore_frac=0.30, slope_threshold=3.0, adaptive_margin=1.0):
    n = len(seq)
    k = max(2, int(np.ceil(n * explore_frac)))
    observed = list(map(float, seq[:k]))
    regime, s = classify_regime(observed, slope_threshold)

    if regime == "Improving":
        obs = observed[:]
        for i in range(k, n):
            cur = float(seq[i])
            rem = n - i - 1
            if rem == 0:
                return cur, i, regime, s

            q = 1 - 1 / max(rem + 1, 2)
            predicted = float(np.quantile(obs, q))
            threshold = max(max(obs), predicted + adaptive_margin)

            if cur >= threshold:
                return cur, i, regime, s

            obs.append(cur)

        return float(seq[-1]), n - 1, regime, s

    if regime == "Deteriorating":
        best = max(observed)
        threshold = max(np.mean(observed), best - 4.0)

        for i in range(k, n):
            cur = float(seq[i])
            if cur >= threshold:
                return cur, i, regime, s

        return float(seq[-1]), n - 1, regime, s

    bench = max(observed)
    for i in range(k, n):
        if seq[i] > bench:
            return float(seq[i]), i, regime, s

    return float(seq[-1]), n - 1, regime, s


def batch_regime(seqs, explore_frac=0.30, slope_threshold=3.0, margin=1.0):
    vals = np.empty(len(seqs))
    pos = np.empty(len(seqs), int)
    regimes = []
    slopes = []

    for j, seq in enumerate(seqs):
        v, p, r, slope_v = regime_aware(
            seq, explore_frac, slope_threshold, margin
        )
        vals[j] = v
        pos[j] = p
        regimes.append(r)
        slopes.append(slope_v)

    return vals, pos, np.array(regimes), np.array(slopes)


# ============================================================
# v4 Game-Fuzzy strategic payoff model
# ============================================================
def strategic_payoffs(
    observed,
    current_score,
    remaining,
    slope_threshold=3.0,
    search_cost=2.0,
    competition=0.25,
    risk_aversion=0.50,
):
    """
    Strategic payoff framing:

    The decision maker chooses STOP or CONTINUE while the unknown future
    environment can be Improving, Stable or Deteriorating.

    This is not claimed to be a Nash equilibrium. The environment is treated
    as uncertain nature, and the model compares expected strategic payoffs.
    """
    beliefs, s, vol, r2 = regime_belief(observed, slope_threshold)
    p = np.array([
        beliefs["Improving"],
        beliefs["Stable"],
        beliefs["Deteriorating"],
    ])

    rem = max(int(remaining), 0)
    horizon = min(rem, 3)
    peak_bonus = vol * math.sqrt(2.0 * math.log(max(rem, 2)))

    improving_future = np.clip(
        current_score + max(s, 0.0) * horizon + peak_bonus, 0, 100
    )
    stable_future = np.clip(
        current_score + 0.15 * peak_bonus, 0, 100
    )
    deteriorating_future = np.clip(
        current_score + min(s, 0.0) * horizon - 0.50 * peak_bonus, 0, 100
    )

    state_future = np.array([
        improving_future,
        stable_future,
        deteriorating_future,
    ])

    effective_search_cost = search_cost * (1.0 + competition)
    uncertainty_penalty = risk_aversion * 0.05 * vol

    continue_payoff = float(p @ state_future) - effective_search_cost
    continue_payoff -= uncertainty_penalty

    stop_payoff = (
        float(current_score)
        - risk_aversion * 0.50
        + competition * 2.0
    )

    delta = stop_payoff - continue_payoff

    # Smooth confidence around the payoff boundary. This is not a Nash
    # equilibrium probability.
    temperature = max(1.0, 2.0 + risk_aversion * 2.0)
    stop_probability = 1.0 / (1.0 + math.exp(-delta / temperature))

    return {
        "beliefs": beliefs,
        "slope": s,
        "volatility": vol,
        "r2": r2,
        "state_future": state_future,
        "stop_payoff": stop_payoff,
        "continue_payoff": continue_payoff,
        "delta": delta,
        "stop_probability": stop_probability,
        "effective_search_cost": effective_search_cost,
    }


def adaptive_warmup_length(seq, explore_frac=0.30):
    """
    v7 adaptive warm-up policy.

    The v6 controller used one fixed exploration length. v7 makes the
    observation window responsive to the early evidence:

      - High early scores -> start deciding earlier so early-surging
        opportunities are not lost.
      - Low, flat/declining early scores -> observe longer before allowing the
        strategic payoff model to stop, reducing premature stops and
        giving late surges time to appear.
      - Otherwise -> retain the v6 baseline exploration fraction.

    This is still a heuristic policy; it does not use future observations.
    """
    n = len(seq)
    base_k = max(2, int(np.ceil(n * explore_frac)))

    if n <= 3:
        return min(base_k, max(1, n - 1))

    first3 = np.asarray(seq[:3], dtype=float)
    early_slope = slope(first3)
    early_mean = float(first3.mean())

    if early_mean >= 75.0:
        return min(max(2, base_k - 1), n - 1)

    if early_mean < 75.0 and early_slope < 3.0:
        return min(base_k + 2, n - 1)

    return min(base_k, n - 1)


def game_fuzzy_stop(
    seq,
    explore_frac=0.30,
    slope_threshold=3.0,
    search_cost=2.0,
    competition=0.25,
    risk_aversion=0.50,
    confidence_gate=0.60,
    advantage_gate=1.0,
):
    """
    v7 Adaptive-Warmup + Confidence-Weighted + Strategy-Advantage policy.

    v7 preserves the v6 strategic payoff model, confidence weighting, and
    bounded strategy-advantage override, but replaces the single fixed
    exploration window with an adaptive warm-up.

    The warm-up is intentionally asymmetric:
      - strong early scores shorten exploration;
      - weak, flat/declining early scores lengthen exploration;
      - mixed evidence keeps the v6 exploration fraction.

    The goal is to reduce premature stops in noisy/late-surge sequences while
    preserving the ability to capture early surges and strong directional
    trends. The policy uses only observations available at decision time.
    """
    n = len(seq)
    k = adaptive_warmup_length(seq, explore_frac)
    observed = list(map(float, seq[:k]))

    for i in range(k, n):
        cur = float(seq[i])
        rem = n - i - 1

        if rem == 0:
            return cur, i

        model = strategic_payoffs(
            observed,
            current_score=cur,
            remaining=rem,
            slope_threshold=slope_threshold,
            search_cost=search_cost,
            competition=competition,
            risk_aversion=risk_aversion,
        )

        conf = regime_confidence(observed, slope_threshold)

        # Classical signal: positive means STOP is attractive.
        classical_signal = cur - max(observed)

        # Strategic signal: positive means STOP is attractive.
        strategic_signal = model["stop_payoff"] - model["continue_payoff"]

        # Reference the best score seen during the exploration prefix.
        classical_reference = max(observed)

        # Strategic continuation advantage over the classical reference.
        # Positive values mean the strategic model sees enough expected future
        # value to justify continuing beyond the classical benchmark.
        strategic_continue_advantage = (
            model["continue_payoff"] - classical_reference
        )

        # Sharper confidence weighting than v5.  The weight rises quickly
        # once confidence crosses the gate, so strong directional regimes do
        # not get unnecessarily diluted by the classical rule.
        low = max(0.0, confidence_gate - 0.10)
        high = min(1.0, confidence_gate + 0.10)
        confidence_weight = float(
            np.clip((conf - low) / max(high - low, 1e-9), 0.0, 1.0)
        )

        # Smoothstep gives a softer transition while still becoming decisive
        # above the confidence gate.
        confidence_weight = (
            confidence_weight
            * confidence_weight
            * (3.0 - 2.0 * confidence_weight)
        )

        # Strategy-advantage override.  This is deliberately bounded so a
        # small noisy advantage cannot overwhelm the confidence mechanism.
        advantage_weight = float(
            np.clip(
                strategic_continue_advantage / max(advantage_gate, 1e-9),
                0.0,
                1.0,
            )
        )

        # Require some evidence of a directional regime before the advantage
        # override can become dominant.  This avoids turning random IID noise
        # into aggressive continuation.
        directional_evidence = float(
            np.clip((conf - 0.30) / 0.40, 0.0, 1.0)
        )

        advantage_override_weight = (
            advantage_weight * directional_evidence
        )

        # Let either strong confidence OR a strong strategic advantage move
        # the policy toward Game-Fuzzy.  Confidence remains the primary signal.
        strategic_weight = max(
            confidence_weight,
            0.75 * advantage_override_weight,
        )

        decision_signal = (
            (1.0 - strategic_weight) * classical_signal
            + strategic_weight * strategic_signal
        )

        # Extra safeguard: if the strategic model has a clear continuation
        # advantage, do not let a tiny positive classical signal force an early
        # stop.  This is the key protection for strong increasing/decreasing
        # trend cases.
        if (
            strategic_continue_advantage >= advantage_gate
            and conf >= 0.35
            and strategic_signal < 0
        ):
            decision_signal = min(decision_signal, strategic_signal)

        if decision_signal >= 0:
            return cur, i

        observed.append(cur)

    return float(seq[-1]), n - 1


def batch_game_fuzzy(
    seqs,
    explore_frac=0.30,
    slope_threshold=3.0,
    search_cost=2.0,
    competition=0.25,
    risk_aversion=0.50,
    confidence_gate=0.60,
    advantage_gate=1.0,
):
    vals = np.empty(len(seqs))
    pos = np.empty(len(seqs), int)

    for j, seq in enumerate(seqs):
        v, p = game_fuzzy_stop(
            seq,
            explore_frac=explore_frac,
            slope_threshold=slope_threshold,
            search_cost=search_cost,
            competition=competition,
            risk_aversion=risk_aversion,
            confidence_gate=confidence_gate,
            advantage_gate=advantage_gate,
        )
        vals[j] = v
        pos[j] = p

    return vals, pos


# ============================================================
# Batch wrappers
# ============================================================
def batch_classical(seqs):
    vals, pos = [], []
    for seq in seqs:
        v, p = classical(seq)
        vals.append(v)
        pos.append(p)
    return np.asarray(vals), np.asarray(pos)


def batch_adaptive(seqs, frac=0.4, margin=1):
    vals, pos = [], []
    for seq in seqs:
        v, p = adaptive(seq, frac, margin)
        vals.append(v)
        pos.append(p)
    return np.asarray(vals), np.asarray(pos)


# ============================================================
# UI
# ============================================================
st.title("🛑 OptiStop — Final Hackathon Dashboard")
st.caption(
    "Fuzzy opportunity scoring + adaptive regime-aware warm-up + strategic stopping under uncertainty"
)

st.markdown(
    """
### Core idea

OptiStop v4 treats stopping as a **strategic decision under uncertainty**.

1. **Fuzzy logic** converts multiple attributes into an opportunity score.
2. **Regime detection** forms a belief over Improving, Stable and Deteriorating.
3. A **strategic payoff model** compares STOP and CONTINUE using future value,
   search cost, competition pressure and uncertainty.
4. When the regime belief is weak, v4 falls back to a classical stopping rule
   rather than pretending the future is known.

**Research note:** the game-theoretic layer is a payoff/game framing, not a
claim of a universal Nash equilibrium or universal optimal policy.
"""
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Live Decision + Fuzzy Logic",
        "Regime Detector",
        "Game-Theoretic Decision",
        "Controlled Benchmark",
        "🏆 Final Dashboard",
    ]
)

# ============================================================
# FINAL HACKATHON DASHBOARD
# ============================================================
with tab5:
    st.header("🏆 Final Hackathon Dashboard")
    st.write(
        "Use this page for the final judge-facing comparison of the "
        "optimal-stopping strategies."
    )
    st.info(
        "Controlled synthetic validation: keep seed, sequence length and "
        "parameters fixed when presenting results."
    )

    a,b,c=st.columns(3)
    with a:
        fd_trials=st.slider("Validation trials",1000,20000,10000,1000,key="fd_trials")
    with b:
        fd_n=st.slider("Opportunities",5,30,10,key="fd_n")
    with c:
        fd_seed=st.number_input("Validation seed",0,1000000,42,key="fd_seed")

    if st.button("🚀 Run final 5-scenario validation",type="primary",key="fd_run"):
        rows=[]
        progress=st.progress(0.0)
        status=st.empty()

        for si,scn in enumerate(SCENARIOS):
            status.write("Running "+scn+"...")
            seq=generate_scenarios(fd_trials,fd_n,fd_seed)[scn]

            vals={k:[] for k in ["Classical","Adaptive","Regime-Aware","Game-Fuzzy v7","Oracle"]}
            regrets=[]
            positions=[]

            for s in seq:
                oracle=float(np.max(s))
                c,cp=classical_secretary(s)
                ad,ap=adaptive_baseline(s,explore_frac=0.40,safety_margin=1.0)
                r,rp,_,_=regime_aware(s,explore_frac=0.30,slope_threshold=3.0,adaptive_margin=1.0)
                g,gp=game_fuzzy_stop(
                    s,explore_frac=0.30,slope_threshold=3.0,
                    search_cost=2.0,competition=0.25,risk_aversion=0.50,
                    confidence_gate=0.60,advantage_gate=1.0
                )

                vals["Classical"].append(c)
                vals["Adaptive"].append(ad)
                vals["Regime-Aware"].append(r)
                vals["Game-Fuzzy v7"].append(g)
                vals["Oracle"].append(oracle)
                regrets.append(oracle-g)
                positions.append(gp+1)

            row={"Scenario":scn}
            for k,v in vals.items():
                row[k]=float(np.mean(v))
            row["v7 vs Classical"]=row["Game-Fuzzy v7"]-row["Classical"]
            row["v7 Regret"]=float(np.mean(regrets))
            row["v7 Stop Position"]=float(np.mean(positions))
            rows.append(row)
            progress.progress((si+1)/len(SCENARIOS))

        st.session_state["final_dashboard_results"]=pd.DataFrame(rows)
        progress.empty()
        status.success("Final validation complete.")

    if "final_dashboard_results" in st.session_state:
        fdf=st.session_state["final_dashboard_results"]

        st.subheader("Final benchmark table")
        st.dataframe(
            fdf.style.format({
                "Classical":"{:.2f}","Adaptive":"{:.2f}",
                "Regime-Aware":"{:.2f}","Game-Fuzzy v7":"{:.2f}",
                "Oracle":"{:.2f}","v7 vs Classical":"{:+.2f}",
                "v7 Regret":"{:.2f}","v7 Stop Position":"{:.2f}"
            }),
            use_container_width=True,hide_index=True
        )

        avg=fdf["Game-Fuzzy v7"].mean()
        base=fdf["Classical"].mean()
        gain=fdf["v7 vs Classical"].mean()
        wins=int((fdf["v7 vs Classical"]>0).sum())

        m1,m2,m3,m4=st.columns(4)
        m1.metric("v7 average score",f"{avg:.2f}")
        m2.metric("Average gain vs Classical",f"{gain:+.2f}")
        m3.metric("Scenario wins",f"{wins}/5")
        m4.metric("Relative gain",f"{100*gain/max(base,1e-9):+.1f}%")

        strategies=["Classical","Adaptive","Regime-Aware","Game-Fuzzy v7"]
        winners=[fdf.loc[i,strategies].idxmax() for i in fdf.index]
        wc=pd.Series(winners).value_counts()
        win_df=pd.DataFrame({
            "Strategy":strategies,
            "Scenario wins":[int(wc.get(s,0)) for s in strategies]
        })
        st.subheader("Scenario wins")
        st.dataframe(win_df,use_container_width=True,hide_index=True)

        avg_df=pd.DataFrame({
            "Strategy":["Classical","Adaptive","Regime-Aware","Game-Fuzzy v7","Oracle"],
            "Average score":[
                fdf["Classical"].mean(),fdf["Adaptive"].mean(),
                fdf["Regime-Aware"].mean(),fdf["Game-Fuzzy v7"].mean(),
                fdf["Oracle"].mean()
            ]
        })
        st.subheader("Average strategy performance")
        st.bar_chart(avg_df.set_index("Strategy"))

        st.subheader("30-second judge explanation")
        st.markdown(
            "**Problem:** when should we stop searching and accept the current "
            "opportunity when future opportunities are uncertain?\n\n"
            "**Optimal stopping** provides the STOP/CONTINUE framework. "
            "**Fuzzy logic** handles noisy opportunity quality and trend evidence. "
            "**Game theory** models search cost, competition and risk. "
            "**Regime awareness** identifies improving, stable and deteriorating "
            "environments. **v7** adapts the warm-up period to early evidence."
        )
        st.warning(
            "These results are from controlled synthetic scenarios. Present them "
            "as simulation evidence, not as proof of universal real-world superiority."
        )

# ------------------------------------------------------------
# Tab 1
# ------------------------------------------------------------
with tab1:
    st.subheader("Fuzzy opportunity scoring")

    a, b, c = st.columns(3)

    with a:
        v = st.slider("Value / compensation", 0, 100, 80, key="live_value")
        fit = st.slider("Personal fit", 0, 100, 85, key="live_fit")

    with b:
        growth = st.slider("Growth potential", 0, 100, 80, key="live_growth")
        org = st.slider("Organization quality", 0, 100, 75, key="live_org")

    with c:
        conv = st.slider("Location / convenience", 0, 100, 70, key="live_convenience")
        risk = st.selectbox(
            "Risk tolerance",
            ["Conservative", "Balanced", "Aggressive"],
        )

    score = fuzzy_score(v, fit, growth, org, conv)
    st.metric("Fuzzy Opportunity Score", f"{score:.1f}/100")

    obs_txt = st.text_input(
        "Observed opportunity scores",
        "62, 71, 68, 76",
    )
    remaining = st.number_input("Opportunities remaining", 0, 100, 5)

    try:
        obs = [float(x.strip()) for x in obs_txt.split(",") if x.strip()]
        if not obs:
            raise ValueError
    except Exception:
        obs = [62, 71, 68, 76]

    regime, s = classify_regime(obs, 3.0)
    st.metric("Detected regime", regime)
    st.write(f"Observed trend slope: **{s:.2f} points/opportunity**")

    margin = {
        "Conservative": 5,
        "Balanced": 1,
        "Aggressive": -2,
    }[risk]

    if remaining == 0:
        decision = "STOP"
        threshold = max(obs)
    elif regime == "Improving":
        arr = np.asarray(obs)
        q = 1 - 1 / (int(remaining) + 1)
        threshold = max(
            arr.max(),
            float(np.quantile(arr, q)) + margin,
        )
        decision = "STOP" if score >= threshold else "CONTINUE"
    elif regime == "Deteriorating":
        threshold = max(np.mean(obs), max(obs) - 4)
        decision = "STOP" if score >= threshold else "CONTINUE"
    else:
        threshold = max(obs)
        decision = "STOP" if score > threshold else "CONTINUE"

    (st.success if decision == "STOP" else st.warning)(
        "🟢 STOP — accept this opportunity"
        if decision == "STOP"
        else "🟡 CONTINUE — keep searching"
    )
    st.info(
        f"Current score={score:.1f}; threshold={threshold:.1f}; regime={regime}."
    )


# ------------------------------------------------------------
# Tab 2
# ------------------------------------------------------------
with tab2:
    st.subheader("Soft regime belief")

    text = st.text_input(
        "Enter observed sequence",
        "62, 64, 60, 66",
    )

    try:
        x = [float(v.strip()) for v in text.split(",") if v.strip()]
        if not x:
            raise ValueError
    except Exception:
        x = [62, 64, 60, 66]

    threshold = st.slider(
        "Slope threshold",
        0.5,
        10.0,
        3.0,
        0.5,
        key="regime_detector_slope_threshold",
    )

    beliefs, s, vol, r2 = regime_belief(x, threshold)
    reg = max(beliefs, key=beliefs.get)

    st.metric("Most likely regime", reg)
    st.metric("Estimated slope", f"{s:.2f}")
    st.metric("Observed volatility", f"{vol:.2f}")
    st.metric("Trend R²", f"{r2:.2f}")

    belief_df = pd.DataFrame(
        {
            "Regime": list(beliefs.keys()),
            "Belief (%)": [100 * value for value in beliefs.values()],
        }
    )
    st.dataframe(belief_df, use_container_width=True)

    st.write(
        "The detector uses only the observations entered; it does not see future "
        "opportunities."
    )


# ------------------------------------------------------------
# Tab 3
# ------------------------------------------------------------
with tab3:
    st.subheader("STOP vs CONTINUE strategic payoff + confidence gate")

    col1, col2 = st.columns(2)

    with col1:
        current_score = st.slider(
            "Current opportunity score",
            0,
            100,
            82,
            key="game_current_score",
        )
        remaining_game = st.number_input(
            "Future opportunities remaining",
            0,
            100,
            5,
        )
        search_cost = st.slider(
            "Search cost",
            0.0,
            10.0,
            2.0,
            0.5,
            key="game_search_cost",
        )

    with col2:
        competition = st.slider(
            "Competition pressure",
            0.0,
            1.0,
            0.25,
            0.05,
            key="game_competition",
        )
        risk_aversion = st.slider(
            "Risk aversion",
            0.0,
            1.0,
            0.50,
            0.05,
            key="game_risk_aversion",
        )
        slope_threshold_game = st.slider(
            "Regime slope threshold",
            0.5,
            10.0,
            3.0,
            0.5,
            key="game_regime_slope_threshold",
        )

    obs_game_txt = st.text_input(
        "Observed history for strategic model",
        "62, 71, 68, 76",
    )

    try:
        obs_game = [
            float(x.strip())
            for x in obs_game_txt.split(",")
            if x.strip()
        ]
        if not obs_game:
            raise ValueError
    except Exception:
        obs_game = [62, 71, 68, 76]

    model = strategic_payoffs(
        obs_game,
        current_score=current_score,
        remaining=int(remaining_game),
        slope_threshold=slope_threshold_game,
        search_cost=search_cost,
        competition=competition,
        risk_aversion=risk_aversion,
    )

    beliefs = model["beliefs"]
    conf = regime_confidence(obs_game, slope_threshold_game)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Improving belief", f"{100*beliefs['Improving']:.1f}%")
    c2.metric("Stable belief", f"{100*beliefs['Stable']:.1f}%")
    c3.metric("Deteriorating belief", f"{100*beliefs['Deteriorating']:.1f}%")
    c4.metric("STOP probability", f"{100*model['stop_probability']:.1f}%")
    c5.metric("Regime confidence", f"{100*conf:.1f}%")

    st.markdown("#### Strategic payoff matrix")

    future = model["state_future"]
    payoff_df = pd.DataFrame(
        {
            "Action": ["STOP", "CONTINUE"],
            "Improving": [
                model["stop_payoff"],
                future[0] - model["effective_search_cost"],
            ],
            "Stable": [
                model["stop_payoff"],
                future[1] - model["effective_search_cost"],
            ],
            "Deteriorating": [
                model["stop_payoff"],
                future[2] - model["effective_search_cost"],
            ],
        }
    )

    st.dataframe(payoff_df.round(2), use_container_width=True)

    p1, p2 = st.columns(2)
    p1.metric("Expected STOP payoff", f"{model['stop_payoff']:.2f}")
    p2.metric("Expected CONTINUE payoff", f"{model['continue_payoff']:.2f}")

    if model["delta"] >= 0:
        st.success(
            f"🟢 STOP — strategic advantage **+{model['delta']:.2f}** points."
        )
    else:
        st.warning(
            f"🟡 CONTINUE — strategic advantage **{model['delta']:.2f}** points."
        )

    st.caption(
        "STOP probability is a smooth decision-confidence measure around the "
        "payoff boundary; it is not presented as a Nash equilibrium."
    )


# ------------------------------------------------------------
# Tab 4
# ------------------------------------------------------------
with tab4:
    st.subheader(
        "Controlled benchmark — identical sequences for every strategy"
    )

    n = st.slider("Opportunities per sequence", 5, 50, 10, key="bench_sequence_length")
    trials = st.slider(
        "Simulation trials",
        1000,
        50000,
        10000,
        step=1000,
        key="bench_trials",
    )

    scenario = st.selectbox(
        "Scenario",
        [
            "Late Surge",
            "Early Surge",
            "IID",
            "Increasing Trend",
            "Decreasing Trend",
        ],
    )

    seed = st.number_input(
        "Random seed",
        0,
        100000,
        42,
    )

    regime_explore = st.slider(
        "Regime exploration fraction",
        0.20,
        0.50,
        0.30,
        0.05,
        key="bench_regime_exploration",
    )

    regime_threshold = st.slider(
        "Regime slope threshold",
        0.5,
        10.0,
        3.0,
        0.5,
        key="bench_regime_slope_threshold",
    )

    adaptive_frac = st.slider(
        "Adaptive exploration fraction",
        0.20,
        0.60,
        0.40,
        0.05,
        key="bench_adaptive_exploration",
    )

    margin = st.slider(
        "Adaptive safety margin",
        -5.0,
        10.0,
        1.0,
        0.5,
        key="bench_adaptive_margin",
    )

    st.markdown("#### v7 adaptive-warmup strategic controls")

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        game_search_cost = st.slider(
            "Search cost",
            0.0,
            10.0,
            2.0,
            0.5,
            key="bench_search_cost",
        )

    with g2:
        game_competition = st.slider(
            "Competition pressure",
            0.0,
            1.0,
            0.25,
            0.05,
            key="bench_competition",
        )

    with g3:
        game_risk = st.slider(
            "Risk aversion",
            0.0,
            1.0,
            0.50,
            0.05,
            key="bench_risk",
        )

    with g4:
        confidence_gate = st.slider(
            "Regime confidence gate",
            0.50,
            0.95,
            0.60,
            0.05,
            key="bench_confidence_gate_v7",
        )

    advantage_gate = st.slider(
        "Strategy advantage gate",
        0.0,
        5.0,
        1.0,
        0.5,
        key="bench_advantage_gate_v7",
    )

    st.caption(
        "v7 keeps v6's confidence-weighted strategic decision layer, but "
        "adapts the warm-up window: strong early opportunities are evaluated "
        "sooner, while weak non-improving openings get more observations before "
        "strategic stopping is allowed."
    )

    if st.button("Run v7 benchmark", type="primary"):
        with st.spinner(
            "Evaluating Classical, Adaptive, Regime-Aware, Game-Fuzzy v7 and Oracle..."
        ):
            seq = generate(
                trials,
                n,
                scenario,
                int(seed),
            )

            oracle = seq.max(1)

            cv, cp = batch_classical(seq)
            av, ap = batch_adaptive(seq, adaptive_frac, margin)
            rv, rp, regs, slopes = batch_regime(
                seq,
                regime_explore,
                regime_threshold,
                margin,
            )
            gv, gp = batch_game_fuzzy(
                seq,
                explore_frac=regime_explore,
                slope_threshold=regime_threshold,
                search_cost=game_search_cost,
                competition=game_competition,
                risk_aversion=game_risk,
                confidence_gate=confidence_gate,
                advantage_gate=advantage_gate,
            )

            result = pd.DataFrame(
                {
                    "Strategy": [
                        "Classical",
                        "Adaptive",
                        "Regime-Aware",
                        "Game-Fuzzy v7",
                        "Oracle (upper bound)",
                    ],
                    "Average selected score": [
                        cv.mean(),
                        av.mean(),
                        rv.mean(),
                        gv.mean(),
                        oracle.mean(),
                    ],
                    "Average regret": [
                        (oracle - cv).mean(),
                        (oracle - av).mean(),
                        (oracle - rv).mean(),
                        (oracle - gv).mean(),
                        0,
                    ],
                    "Best selected (%)": [
                        100 * np.mean(cv == oracle),
                        100 * np.mean(av == oracle),
                        100 * np.mean(rv == oracle),
                        100 * np.mean(gv == oracle),
                        100,
                    ],
                    "Avg stop position": [
                        cp.mean() + 1,
                        ap.mean() + 1,
                        rp.mean() + 1,
                        gp.mean() + 1,
                        np.argmax(seq, 1).mean() + 1,
                    ],
                }
            )

        st.dataframe(result.round(4), use_container_width=True)

        improvement = gv.mean() - cv.mean()
        if improvement > 0:
            st.success(
                f"Game-Fuzzy v7 vs Classical: **+{improvement:.2f} points** "
                "average selected score."
            )
        else:
            st.warning(
                f"Game-Fuzzy v7 vs Classical: **{improvement:.2f} points** "
                "average selected score."
            )

        st.info(
            f"Regime-Aware v3 vs Classical: **{rv.mean() - cv.mean():+.2f} points**."
        )

        counts = pd.Series(regs).value_counts().reindex(
            ["Improving", "Stable", "Deteriorating"],
            fill_value=0,
        )

        st.markdown("### Detected regimes in this benchmark")
        st.dataframe(
            pd.DataFrame(
                {
                    "Regime": counts.index,
                    "Trials": counts.values,
                    "Share (%)": 100 * counts.values / trials,
                }
            ),
            use_container_width=True,
        )

        # Estimate how often the confidence gate would be considered strong
        # using the same observed-prefix information used by v5.
        conf_sample = np.empty(trials)
        warmup_sample = np.empty(trials)
        for j in range(trials):
            kk = adaptive_warmup_length(seq[j], regime_explore)
            warmup_sample[j] = kk
            conf_sample[j] = regime_confidence(
                seq[j, :kk],
                regime_threshold,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Mean regime confidence",
            f"{100 * conf_sample.mean():.1f}%",
        )
        m2.metric(
            "High-confidence share",
            f"{100 * np.mean(conf_sample >= confidence_gate):.1f}%",
        )
        m3.metric(
            "Avg adaptive warm-up",
            f"{warmup_sample.mean():.2f} opportunities",
        )

        st.line_chart(
            pd.DataFrame(
                {"Example opportunity score": seq[0]},
                index=np.arange(1, n + 1),
            )
        )

        st.caption(
            "All five strategies use the exact same sequences. Oracle is an "
            "upper bound, not a deployable strategy. The synthetic benchmark "
            "uses raw opportunity scores; fuzzy multi-attribute scoring is "
            "demonstrated in the live decision tab."
        )


st.divider()
st.caption(
    "Research prototype. Not for high-stakes decisions. v7 adds adaptive warm-up on top of v6's sharper "
    "confidence weighting and a bounded strategic-advantage override; it does "
    "not claim a universal optimal policy."
)
