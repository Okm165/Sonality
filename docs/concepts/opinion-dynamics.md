# Opinion Dynamics

Sonality's belief system draws from formal opinion dynamics models — mathematical frameworks that describe how agents update their beliefs in response to social influence and evidence. This page explains the magnitude formula, Bayesian belief resistance, power-law decay, and structural disagreement detection, with research grounding from Friedkin-Johnsen, Hegselmann-Krause, Deffuant, and Oravecz et al.

## The Magnitude Formula

When an interaction passes the ESS threshold (`ess.score > 0.3`), the raw update magnitude is:

$$\text{magnitude} = \text{OPINION\_BASE\_RATE} \times \text{ess.score} \times \max(\text{ess.novelty}, 0.1) \times \text{dampening}$$

| Factor | Value | Purpose |
|--------|-------|---------|
| `OPINION_BASE_RATE` | 0.1 | Conservative per-update step; caps maximum shift at 10% of opinion space |
| `ess.score` | 0.0–1.0 | ESS argument quality |
| `max(ess.novelty, 0.1)` | 0.1–1.0 | Diminishing returns on repeated arguments; floor prevents zero magnitude |
| `dampening` | 0.5 or 1.0 | 0.5× for first 10 interactions (`interaction_count < BOOTSTRAP_DAMPENING_UNTIL`) |

**Bootstrap dampening** maps to the Deffuant model's "initial uncertainty" concept.[^1] Without it, the agent's personality would be disproportionately shaped by whoever talks to it first — "first-impression dominance" (Chameleon LLMs, EMNLP 2025).

**Example magnitudes:**

| Scenario | Magnitude | Percentage |
|----------|-----------|------------|
| Maximum (mature, score=1.0, novelty=1.0) | 0.100 | 10.0% |
| Typical high-ESS (score=0.7, novelty=0.6) | 0.042 | 4.2% |
| Bootstrap (score=0.7, novelty=0.6, first 10) | 0.021 | 2.1% |
| Minimum meaningful (score=0.31, novelty=0.1) | 0.003 | 0.3% |

## Bayesian Belief Resistance

Established beliefs resist change proportionally to their evidence base. Each belief's `BeliefMeta` tracks a confidence score that grows logarithmically:

$$\text{confidence} = \frac{\log_2(\text{evidence\_count} + 1)}{\log_2(20)}$$

Capped at 1.0. The effective magnitude is:

$$\text{effective\_magnitude} = \frac{\text{magnitude}}{\text{confidence} + 1.0}$$

| Evidence Count | Confidence | Effective Mag (raw 0.04) |
|----------------|------------|---------------------------|
| 1 | ≈ 0.23 | 0.033 (≈ 0.81× base) |
| 5 | ≈ 0.58 | 0.025 (≈ 0.63× base) |
| 10 | ≈ 0.76 | 0.023 (≈ 0.57× base) |
| 19 | 1.00 | 0.020 (0.5× base) |

!!! tip "The Math"
    With `conf = 0.58`: `effective_mag = 0.04 / 1.58 ≈ 0.025`. With `conf = 1.0`: `effective_mag = 0.04 / 2.0 = 0.020`. A belief backed by 10 conversations is roughly 1.5× harder to shift than a new belief — proportional resistance, not immunity.

This implements sequential Bayesian updating (Oravecz et al., 2016)[^2] — posterior distributions serve as priors for the next update. Hegselmann-Krause (2002)[^3]: only sufficiently strong evidence should shift opinions.

**Structural disagreement bonus**: When the user argues *against* the agent's existing stance (`position × direction < 0`), the code adds `abs(old_pos)` to the confidence denominator, further resisting flip-flopping.

## Opinion Vector Updates

$$\text{new} = \text{clamp}(\text{old} + \text{direction} \times \text{effective\_magnitude}, -1.0, 1.0)$$

- `direction` = +1.0 (supports), -1.0 (opposes), or 0.0 (neutral; no update)
- Clamping keeps opinions in [-1, 1]

## Power-Law Belief Decay

During reflection only, unreinforced beliefs lose confidence following a power-law retention curve:

$$R(t) = (1 + \text{gap})^{-\beta}$$

Where:

- `gap` = `interaction_count - last_reinforced` (interactions since belief was last reinforced)
- β = 0.15 (`BELIEF_DECAY_RATE`, from FadeMem 2026 / Ebbinghaus-inspired curves)
- Decay runs **only** on beliefs with `gap ≥ 5`

**Reinforcement floor** prevents well-evidenced beliefs from decaying to nothing:

$$\text{floor} = \min(0.6, \text{evidence\_count} \times 0.06)$$

$$\text{new\_conf} = \max(\text{floor}, \text{conf} \times R(t))$$

**Minimum confidence threshold**: 0.05. Below this, the belief is dropped entirely (removed from `opinion_vectors` and `belief_meta`).

| Gap (interactions) | Retention (β=0.15) | With floor (ev=5, floor=0.30) |
|-------------------|--------------------|-------------------------------|
| 5 | 0.78 | max(0.30, conf × 0.78) |
| 10 | 0.69 | max(0.30, conf × 0.69) |
| 20 | 0.61 | max(0.30, conf × 0.61) |
| 50 | 0.49 | max(0.30, conf × 0.49) |

Power-law (not exponential) matches the Ebbinghaus human memory curve and neural network forgetting research.[^4]

## Structural Disagreement Detection

Sonality detects disagreement structurally rather than via keywords:

```python
# User argued against agent's existing stance
for topic in ess.topics:
    position = sponge.opinion_vectors.get(topic, 0.0)
    if abs(position) > 0.1 and position * ess_direction < 0:
        return True  # disagreement detected
```

Keyword approaches ("I disagree", "I don't think") miss nuanced, qualified, or reframing-based disagreements. The CARE framework (EMNLP 2025) documents "conceptual gap" and "reasoning gap" for keyword-based detection.

The disagreement rate is tracked as a running mean in `behavioral_signature.disagreement_rate`. Target: 20–35% (DEBATE benchmark human baselines). Below 15% suggests sycophancy; above 50% suggests contrarianism.

## Connection to Friedkin-Johnsen Model

The Friedkin-Johnsen (FJ) model is the standard mathematical framework for opinion dynamics. In FJ, each agent has an innate opinion \( s_i \), a stubbornness parameter \( \lambda_i \in [0,1] \), and a trust matrix \( T \). The update rule:

$$x_i(t+1) = \lambda_i \cdot s_i + (1 - \lambda_i) \cdot \sum T_{ij} \cdot x_j(t)$$

Sonality maps onto this framework:

| FJ Variable | Sonality Equivalent | Implementation |
|-------------|---------------------|----------------|
| \( s_i \) (innate opinion) | `CORE_IDENTITY` + `SEED_SNAPSHOT` | Immutable anchor against drift |
| \( \lambda_i \) (stubbornness) | `1 / (confidence + 1)` scaling | Higher evidence = more stubborn |
| \( T_{ij} \) (trust weight) | ESS score × novelty | Higher argument quality = more trust |

Research finding (arXiv:2410.22577): moderate stubbornness in neutral agents *reduces* polarization — counterintuitive but validated. Sonality's initial agent is neutral; Bayesian resistance provides moderate stubbornness that increases with evidence, naturally avoiding both excessive volatility and excessive rigidity.

The Diminishing Stubbornness Extension (arXiv:2409.12601) shows that stubbornness that decreases over time leads to eventual convergence. Sonality achieves this implicitly: bootstrap dampening adds extra stubbornness early (0.5× scaling), then relaxes. As the agent matures, novelty scores for established topics decrease, producing smaller updates — adaptive stubbornness without explicit scheduling.

## Research Grounding

| Source | Key Finding |
|--------|-------------|
| **Friedkin-Johnsen** | Stubbornness balancing initial beliefs vs social influence; Sonality's Bayesian resistance maps to λ |
| **Hegselmann-Krause (2002)** | Bounded confidence — only sufficiently strong evidence shifts opinions; ESS threshold implements this |
| **Deffuant model** | Initial uncertainty, convergence dynamics; bootstrap dampening prevents first-impression dominance |
| **Oravecz et al. (2016)** | Sequential Bayesian personality assessment; posterior-as-prior for next update |
| **AGM framework** | Belief revision consistency requirements |
| **Neural Howlround (arXiv:2504.07992)** | Self-reinforcing cognitive loops; 67% of conversations; resistance mechanisms counter this |
| **Stubbornness Reduces Polarization (2024)** | Moderate stubbornness in neutral agents reduces polarization |
| **Diminishing Stubbornness (2024)** | Decreasing stubbornness over time leads to convergence |

---

**Next:** [Reflection](reflection.md) — when and how belief decay runs. [Personality Development](../personality-development.md) — expected opinion dynamics at each interaction milestone.

[^1]: Deffuant model — initial uncertainty, convergence dynamics.
[^2]: Oravecz et al. (2016). Sequential Bayesian personality assessment.
[^3]: Hegselmann-Krause (2002). Bounded confidence model.
[^4]: Ebbinghaus in LLMs (2025); FadeMem 2026 (arXiv:2601.18642).
