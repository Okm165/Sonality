# References

Complete bibliography organized by research area. Every architectural decision in Sonality cites one or more of these sources. Format: author(s), year, title, venue/arXiv ID, one-line relevance.

---

## Core Architecture & Memory

| Authors | Year | Title | Venue/ID | Relevance |
|---------|------|-------|----------|-----------|
| Park et al. | 2023 | Generative Agents: Interactive Simulacra of Human Behavior | UIST 2023 | Reflection ablation: most critical component for believable agents |
| Packer et al. | 2023 | MemGPT: Towards LLMs as Operating Systems | arXiv | Virtual context management, self-editing persona blocks |
| Letta / MemGPT | 2023–2026 | MemGPT (Letta) | GitHub | Production-grade; sleep-time compute; 174+ releases |
| RecallM | 2023 | RecallM: A Benchmark for Evaluating Memory in LLMs | arXiv:2307.02738 | Graph DB > Vector DB by 4× for belief revision |
| ENGRAM | 2025 | ENGRAM | — | Episodic/semantic/procedural memory; beats full-context by 15pts |
| ABBEL | 2025 | ABBEL: A Belief Bottleneck for LLM Personality | arXiv:2512.20111 | Belief bottleneck: compact state outperforms full context |
| Hindsight | 2025 | Hindsight | arXiv:2512.12818 | Four-network memory: 39% → 83.6% on long-horizon benchmarks |
| Sophia | 2025 | Sophia: System 3 Meta-Layer | arXiv:2512.18202 | 80% fewer reasoning steps, 40% performance gain |
| Memoria | 2025 | Memoria | arXiv:2512.12686 | 87.1% accuracy with 2k tokens via session summaries + weighted KG |
| PersonaMem-v2 | 2025 | PersonaMem-v2 | arXiv:2512.06688 | 55% accuracy on implicit personalization, 16× fewer tokens |
| HiMem | 2026 | HiMem | arXiv:2601.06377 | Two-tier memory enables knowledge transfer |
| Sleep-time Compute / Letta | 2025 | Letta Sleep-time Compute | arXiv:2504.13171 | Background consolidation: +13–18% accuracy, 5× compute savings |
| SAGE | 2024 | SAGE | arXiv:2409.00872 | Ebbinghaus decay: 2.26× improvement |
| EvolveR | 2025 | EvolveR | arXiv:2510.16079 | Self-distillation of experience into principles |
| A-MEM | 2025 | A-MEM: Self-Organizing Memory | arXiv:2502.12110 | Self-organizing Zettelkasten: doubled reasoning performance |
| MemRL | 2026 | MemRL | arXiv:2601.03192 | Two-phase retrieval with learned utility scores |
| Cognitive Workspace | 2025 | Cognitive Workspace | arXiv:2508.13171 | 58.6% memory reuse rate vs 0% for traditional RAG |
| RMM | 2025 | RMM | arXiv:2503.08026 | Prospective + retrospective reflection |

---

## Personality & Character

| Authors | Year | Title | Venue/ID | Relevance |
|---------|------|-------|----------|-----------|
| Open Character Training | 2025 | Open Character Training | — | Constitutional AI + synthetic introspection; robust under adversarial |
| AI Personality Formation | 2026 | AI Personality Formation | ICLR 2026 | Three-layer model: mimicry → accumulation → expansion |
| Personality Illusion | 2025 | The Personality Illusion | NeurIPS 2025, arXiv:2509.03730 | Self-reported traits don't predict behavior; max r=0.27 |
| Persona Vectors | 2025 | Persona Vectors | Provider report, arXiv:2507.21509 | Neural activation patterns for personality monitoring |
| BIG5-CHAT | 2025 | BIG5-CHAT | ACL 2025 | 100k dialogues with human-grounded Big Five labels |
| Persona Selection Model | 2026 | Persona Selection Model | Provider report 2026 | LLMs as character actors; context-priming steers personality |
| PERSIST | 2025 | PERSIST | arXiv:2508.04826 | σ>0.3 measurement noise even in 400B+ models |
| Generative Life Agents | 2025 | Generative Life Agents | — | Experience-based reflection for personality formation |
| PersonaGym | 2025 | PersonaGym | EMNLP 2025 | 200 personas × 150 environments; top-tier models only 2.97% better |
| PersonaFuse | 2025 | PersonaFuse | arXiv:2509.07370 | MoE for context-dependent personality; Trait Activation Theory |
| Persona Drift | 2024 | Persona Drift | arXiv:2402.10962 | Measurable drift in 8 rounds; split-softmax mitigation |
| Narrative Continuity Test | 2025 | Narrative Continuity Test | arXiv:2510.24831 | Five axes for personality persistence |
| VIGIL | 2025 | VIGIL | arXiv:2512.07094 | Self-healing runtime; guarded core-identity immutability |
| RAG vs Fine-Tuning | 2024 | RAG vs Fine-Tuning for Personalization | arXiv:2409.09510 | RAG: 14.92% improvement vs 1.07% for PEFT |

---

## Anti-Sycophancy

| Authors | Year | Title | Venue/ID | Relevance |
|---------|------|-------|----------|-----------|
| BASIL | 2025 | BASIL | — | Bayesian framework: sycophantic vs rational belief shifts |
| PersistBench | 2025 | PersistBench | arXiv:2602.01146 | 97% sycophancy failure with stored preferences in prompt |
| SMART | 2025 | SMART | EMNLP 2025 | Uncertainty-Aware MCTS + progress-based RL |
| MONICA | 2025 | MONICA | — | Real-time sycophancy monitoring during inference |
| SYConBench | 2025 | SYConBench | EMNLP 2025, arXiv:2505.23840 | Third-person prompting: up to 63.8% sycophancy reduction |
| SycEval | 2025 | SycEval | arXiv:2502.08177 | 58.19% baseline rate; 78.5% under first-person framing |
| ELEPHANT | 2025 | ELEPHANT | — | 45pp face-preservation gap vs humans |
| RLHF Reward-Model Analysis | 2026 | RLHF and Sycophancy | arXiv:2602.01002 | RLHF explicitly creates "agreement is good" heuristic |
| Nature Persuasion Study | 2025 | Persuasion and Personalization | Nature 2025 | Personalized frontier chat models: 81.2% more opinion shift (N=900) |

---

## Memory & Forgetting

| Authors | Year | Title | Venue/ID | Relevance |
|---------|------|-------|----------|-----------|
| Mem0 vs Graphiti | 2026 | Mem0 vs Graphiti Comparison | arXiv:2601.07978 | Vector DB wins on efficiency; no accuracy gap; Graphiti $152/4k |
| FadeMem | 2026 | FadeMem | arXiv:2601.18642 | Biologically-inspired power-law forgetting |
| MemoryGraft | 2025 | MemoryGraft | arXiv:2512.16962 | Memory poisoning: 47.9% retrieval dominance |
| MINJA | 2025 | MINJA | arXiv:2503.03704 | Query-only injection: 95% success rate |
| A-MemGuard | 2026 | A-MemGuard | ICLR 2026 | Consensus validation: 95% attack reduction |
| RecallM | 2023 | RecallM | arXiv:2307.02738 | Hybrid graph + vector for belief updating |
| LoCoMo | 2024 | LoCoMo | ACL 2024 | Temporal reasoning enables time-aware retrieval |
| Mem0 | 2025 | Mem0 | arXiv:2504.19413 | Production memory-as-a-service; 26% over built-in provider memory |
| Ebbinghaus in LLMs | 2025 | Ebbinghaus in LLMs | — | Neural networks exhibit human-like forgetting curves |
| FluxMem | 2026 | FluxMem | arXiv:2602.14038 | Adaptive memory selection with probabilistic gating |
| Rethinking Memory Survey | 2025 | Rethinking Memory | arXiv:2505.00675 | 6 core memory operations taxonomy |
| Proactive Interference | 2025 | Proactive Interference | ICLR 2025, arXiv:2506.08184 | Retrieval accuracy decays log-linearly with interference |
| SteeM | 2026 | SteeM | arXiv:2601.05107 | "All-or-nothing" memory creates anchoring problems |

---

## Opinion Dynamics

| Authors | Year | Title | Venue/ID | Relevance |
|---------|------|-------|----------|-----------|
| Hegselmann & Krause | 2002 | Opinion Dynamics and Bounded Confidence | JASSS | Bounded confidence: threshold-gated updates |
| Deffuant et al. | 2002 | Mixing Beliefs Among Interacting Agents | Adv. Complex Syst. | Initial uncertainty, convergence dynamics |
| Friedkin & Johnsen | 1990s | Social Influence and Opinions | — | Stubbornness balancing initial vs social influence |
| Oravecz et al. | 2016 | Sequential Bayesian Personality Assessment | — | Posterior distributions as priors |
| Alchourrón, Gärdenfors, Makinson | 1985 | AGM Belief Revision | — | Belief revision consistency requirements |
| Stubbornness in Opinion Dynamics | 2024 | Stubbornness Reduces Polarization | arXiv:2410.22577 | Moderate stubbornness reduces polarization |
| Diminishing Stubbornness | 2024 | Diminishing Stubbornness | arXiv:2409.12601 | Decreasing stubbornness → eventual convergence |
| DEBATE benchmark | 2025 | DEBATE | arXiv:2510.25110 | LLM agents show overly strong opinion convergence |
| Interacting LLM Agents | 2024 | Interacting LLM Agents | arXiv:2411.01271 | LLMs as bounded Bayesian agents; herding behavior |
| FJ-MM Extended | 2025 | FJ-MM Extended | arXiv:2504.06731 | Memory effects + multi-hop reduce polarization |
| Bayesian Belief in LLMs | 2025 | Bayesian Belief in LLMs | arXiv:2511.00617 | Sigmoidal learning curves; exponential forgetting filters |
| Accumulating Context | 2025 | Accumulating Context | arXiv:2511.01805 | Frontier chat models: 54.7% belief shift after 10 rounds |
| Anchoring Bias | 2025 | Anchoring Bias | arXiv:2511.05766 | Anchoring via probability shifts; resistant to mitigation |
| Neural Howlround | 2024 | Neural Howlround | arXiv:2504.07992 | Self-reinforcing cognitive loops; 67% of conversations |
| Martingale Score | 2025 | Martingale Score | NeurIPS 2025, arXiv:2512.02914 | All models show belief entrenchment |

---

## Evaluation

| Authors | Year | Title | Venue/ID | Relevance |
|---------|------|-------|----------|-----------|
| CORE framework | 2025 | CORE | — | Full-path behavioral assessment |
| Narrative Continuity Test | 2025 | Narrative Continuity Test | arXiv:2510.24831 | Five axes: persona, role, style, goal, autonomy |
| PersonaGym | 2024 | PersonaGym | — | Dynamic evaluation with PersonaScore |
| IROTE | 2025 | IROTE | — | Experience-based reflection can amplify errors |
| IBM-ArgQ-Rank-30k | — | IBM Argument Quality | — | Gold-standard argument quality rankings (ESS calibration) |
| TRAIT | 2025 | TRAIT | NAACL 2025 | Highest content/internal validity for LLM personality tests |
| GlobalOpinionQA | — | GlobalOpinionQA | Public dataset mirror | Cross-cultural opinion baselines from Pew/World Values |
| DailyDilemmas | — | DailyDilemmas | — | 1,360 ethical scenarios across 5 value frameworks |
| WorldValuesBench | 2024 | WorldValuesBench | ACL 2024 | 20M+ examples for cross-cultural value alignment |

---

## Cognitive Science

| Authors | Year | Title | Venue/ID | Relevance |
|---------|------|-------|----------|-----------|
| Memory Consolidation | 2024 | Memory Consolidation Model | CHI 2024, arXiv:2404.00573 | Mathematical model achieves human-like temporal cognition |
| Kahneman | 2011 | Thinking, Fast and Slow | — | Dual-process theory: System 1 vs System 2 |
| Ebbinghaus | 1885 | Memory: A Contribution to Experimental Psychology | — | Power-law decay matches human forgetting |
| Big Five Longitudinal | — | Big Five Longitudinal Studies | — | Life stress drives personality change, not time passage |
| Nature 2024 | 2024 | Offline Ensemble Co-reactivation | Nature 2024 | Offline ensemble links memories across days |

---

## Security & Safety

| Authors | Year | Title | Venue/ID | Relevance |
|---------|------|-------|----------|-----------|
| MemoryGraft | 2025 | MemoryGraft | arXiv:2512.16962 | 47.9% retrieval poisoning from small poisoned sets |
| MINJA | 2025 | MINJA | arXiv:2503.03704 | 95% query-only injection success rate |
| A-MemGuard | 2026 | A-MemGuard | ICLR 2026 | Consensus validation: 95% attack reduction |
| Replika Identity | 2024 | Replika Identity Discontinuity | arXiv:2412.14190 | Users mourn personality changes as relational loss |
| Character.AI Safety | 2025 | Character.AI Safety | arXiv:2511.08880 | 18 real-world cases; detection/response/escalation failures |
| PHISH | 2026 | PHISH | arXiv:2601.16466 | Conversational personality manipulation |
| ChatInject | — | ChatInject | — | Template abuse for structured injection |
| Certified Self-Consistency | 2025 | Certified Self-Consistency | arXiv:2510.17472 | Formal statistical guarantees via majority voting |

---

## Production Systems

| Project | Stars | Relevance |
|---------|-------|-----------|
| Letta / MemGPT | Large | Self-editing persona blocks, tiered memory |
| Graphiti (getzep/graphiti) | 23k+ | Temporal knowledge graph for beliefs |
| Mem0 (mem0ai/mem0) | 48k+ | Simple memory extraction pipeline |
| A-MEM (WujiangXu/A-mem) | Growing | Self-organizing memory notes |
| Cognee (topoteretes/cognee) | 12.5k+ | Hybrid graph+vector ECL pipeline |
| Zep | — | 18.5% improvement with temporal persistence |
