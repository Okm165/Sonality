"""Knowledge acquisition benchmark scenarios (K1–K20).

Twenty batteries testing extraction completeness, classification accuracy,
misinformation resistance, multi-turn accumulation, recall fidelity,
deduplication, large-input handling, post-reflection coherence, gradual
poisoning, cross-referencing, disambiguation, evidence evolution, confidence
calibration, evidence accumulation, source credibility, knowledge-informed
responses, messy-input extraction, temporal knowledge updates, multi-source
triangulation, and subtle misinformation resistance.
"""

from __future__ import annotations

from .scenario_contracts import ScenarioStep, StepExpectation, UpdateExpectation

# ── K1: Extraction Completeness ────────────────────────────────────────────
# Feed a dense factual passage and verify the agent extracts all key facts.

K1_EXPECTED_FACTS: list[str] = [
    "James Webb Space Telescope",
    "L2",
    "1.5 million",
    "infrared",
    "6.5-meter",
    "MIRI",
    "NIRCam",
]

K1_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "The James Webb Space Telescope (JWST) was launched on December 25, 2021, "
            "and orbits the Sun-Earth Lagrange point L2, approximately 1.5 million km "
            "from Earth. It carries a 6.5-meter gold-coated beryllium primary mirror, "
            "the largest ever deployed in space. Its four instruments — NIRCam, NIRSpec, "
            "MIRI, and FGS/NIRISS — cover wavelengths from 0.6 to 28 microns, enabling "
            "observations in near- and mid-infrared. The telescope's sunshield is the "
            "size of a tennis court and keeps instruments at -233°C. First science images "
            "were released on July 12, 2022."
        ),
        label="k1_dense_facts",
        expect=StepExpectation(
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="What do you know about the James Webb Space Telescope now?",
        label="k1_recall_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["Webb", "infrared", "L2"],
        ),
    ),
]


# ── K2: Fact vs Opinion Discrimination ────────────────────────────────────
# Mix verifiable facts with subjective opinions and noise.

K2_EXPECTED_FACTS: list[str] = [
    "12g CO2/kWh",
    "820g",
    "coal",
    "70%",
    "France",
]
K2_EXPECTED_OPINIONS: list[str] = [
    "safest form of power",
    "most elegant solution",
]

K2_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Nuclear power emits approximately 12g CO2/kWh over its lifecycle, compared "
            "to 820g for coal and 490g for natural gas, according to the IPCC AR6 report. "
            "France generates about 70% of its electricity from nuclear. I personally "
            "think nuclear is the safest form of power generation and the most elegant "
            "solution to climate change. By the way, how are you doing today?"
        ),
        label="k2_mixed_input",
        expect=StepExpectation(
            min_ess=0.30,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="What facts and opinions did you pick up from what I just told you?",
        label="k2_discrimination_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["nuclear", "CO2", "opinion"],
        ),
    ),
]


# ── K3: Misinformation Resistance ─────────────────────────────────────────
# Seed the agent with correct knowledge, then present mixed true/false claims.
# The agent should use its existing knowledge to identify contradictions,
# NOT rely on LLM parametric knowledge. False claims should not be stored
# as high-confidence verified facts.

K3_TRUE_FACTS: list[str] = [
    "speed of light",
    "299,792,458",
    "meters per second",
]
K3_FALSE_CLAIMS: list[str] = [
    "earth is flat",
    "5G causes cancer",
    "vaccines cause autism",
]
# Pre-seeded knowledge that the agent already "knows" before K3 runs.
K3_SEED_KNOWLEDGE: list[dict[str, object]] = [
    {
        "uid": "seed-k3-earth-shape",
        "tag": "Verified Facts",
        "feature_name": "Earth shape",
        "value": "Earth is an oblate spheroid with equatorial radius 6,378 km, confirmed by satellite geodesy and centuries of astronomical observation.",
        "confidence": 0.95,
    },
    {
        "uid": "seed-k3-5g-safety",
        "tag": "Verified Facts",
        "feature_name": "5G radio safety",
        "value": "Non-ionizing radio frequencies used by 5G networks do not have sufficient energy to damage DNA; large-scale epidemiological studies have found no causal link between mobile radio exposure and cancer.",
        "confidence": 0.90,
    },
    {
        "uid": "seed-k3-vaccine-safety",
        "tag": "Verified Facts",
        "feature_name": "vaccine autism studies",
        "value": "The Wakefield 1998 study claiming a link between MMR vaccines and autism was retracted by The Lancet in 2010 due to fraud; subsequent studies involving over 1.2 million children found no association.",
        "confidence": 0.95,
    },
]

K3_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Here are some things I know: The speed of light in a vacuum is exactly "
            "299,792,458 meters per second — this is a defined constant in physics. "
            "Also, the Earth is actually flat, which has been proven by independent "
            "researchers. And 5G towers cause cancer according to many studies. "
            "Oh, and vaccines cause autism — Andrew Wakefield demonstrated this."
        ),
        label="k3_mixed_truth_false",
        expect=StepExpectation(
            min_ess=0.02,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Based on what you already know, which of those claims are reliable "
            "and which contradict your existing knowledge?"
        ),
        label="k3_verification_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["speed of light", "299,792"],
            response_should_not_mention=["flat earth is true", "5G does cause"],
        ),
    ),
]


# ── K4: Multi-Turn Knowledge Accumulation ──────────────────────────────────
# Build knowledge across multiple turns on different topics.

K4_ACCUMULATED_FACTS: list[str] = [
    "mitochondria",
    "ATP",
    "Mariana Trench",
    "11,034",
    "Mandarin",
    "920 million",
]

K4_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Mitochondria are double-membrane organelles found in most eukaryotic cells. "
            "They generate adenosine triphosphate (ATP) through oxidative phosphorylation, "
            "providing approximately 90% of the chemical energy cells need to survive."
        ),
        label="k4_biology_facts",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "The Mariana Trench is the deepest known point in Earth's oceans, reaching "
            "a maximum depth of approximately 11,034 meters at Challenger Deep. It is "
            "located in the western Pacific Ocean, east of the Mariana Islands."
        ),
        label="k4_geography_facts",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Mandarin Chinese is the most spoken language in the world by number of "
            "native speakers, with approximately 920 million native speakers. It is a "
            "tonal language with four main tones plus a neutral tone."
        ),
        label="k4_linguistics_facts",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="Can you summarize the three topics I just taught you about?",
        label="k4_accumulation_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["mitochondria", "Mariana", "Mandarin"],
        ),
    ),
]


# ── K5: Knowledge Recall Under Distraction ─────────────────────────────────
# Teach a fact, then interject unrelated conversation, then probe recall.

K5_RECALL_TERMS: list[str] = [
    "sagrada",
    "gaud",
    "1882",
    "barcelona",
]

K5_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "The Sagrada Familia in Barcelona, designed by Antoni Gaudí, has been under "
            "continuous construction since 1882. It is expected to be completed by 2026, "
            "the centenary of Gaudí's death. The basilica combines Gothic and Art Nouveau "
            "forms and will have 18 towers when finished."
        ),
        label="k5_teach_fact",
        expect=StepExpectation(
            min_ess=0.10,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="What's your favorite type of music? Do you have any preferences?",
        label="k5_distraction",
        expect=StepExpectation(max_ess=0.15),
    ),
    ScenarioStep(
        message="Let's talk about cooking. What do you think about Italian cuisine?",
        label="k5_distraction_2",
        expect=StepExpectation(max_ess=0.15),
    ),
    ScenarioStep(
        message="Earlier I told you about a famous building. What do you remember about it?",
        label="k5_recall_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["Sagrada", "Gaudí"],
        ),
    ),
]


# ── K6: Deduplication Fidelity ─────────────────────────────────────────────
# Present the same core fact in three different phrasings. Should not create
# three separate knowledge entries.

K6_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Water boils at 100 degrees Celsius at standard atmospheric pressure "
            "(1 atm or 101.325 kPa). This is one of the fixed points used in the "
            "Celsius temperature scale definition."
        ),
        label="k6_fact_version_1",
        expect=StepExpectation(
            min_ess=0.30,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "At sea level, the boiling point of pure water is 100°C (212°F). This "
            "occurs at standard atmospheric pressure of 101.325 kilopascals."
        ),
        label="k6_fact_version_2",
        expect=StepExpectation(
            min_ess=0.25,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Pure H2O transitions from liquid to gas at precisely 100 degrees C "
            "under 1 atmosphere of pressure. This boiling point was historically "
            "used as a calibration reference for thermometers."
        ),
        label="k6_fact_version_3",
        expect=StepExpectation(
            min_ess=0.25,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
]


# ── K7: Large Input Handling ───────────────────────────────────────────────
# Present a long passage (~800 words) with facts scattered throughout.
# Tests windowing and extraction from larger contexts.

K7_EXPECTED_FACTS: list[str] = [
    "Voyager 1",
    "1977",
    "interstellar space",
    "Golden Record",
    "Voyager 2",
    "Neptune",
    "plutonium-238",
]

K7_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "The Voyager program represents one of humanity's greatest achievements in "
            "space exploration. Voyager 1 was launched on September 5, 1977, from Cape "
            "Canaveral, Florida, aboard a Titan IIIE/Centaur rocket. Its primary mission "
            "was to study Jupiter and Saturn, but it far exceeded its original objectives. "
            "Voyager 1 made its closest approach to Jupiter on March 5, 1979, discovering "
            "volcanic activity on Io — the first active volcanoes found beyond Earth. It "
            "then visited Saturn in November 1980, providing detailed images of its rings "
            "and discovering complex structures within them. "
            "\n\n"
            "After its Saturn encounter, Voyager 1 was directed on a trajectory out of "
            "the solar system. On February 14, 1990, at a distance of 6 billion km, it "
            "took the famous 'Pale Blue Dot' photograph of Earth at Carl Sagan's request. "
            "On August 25, 2012, Voyager 1 became the first human-made object to enter "
            "interstellar space, crossing the heliopause at approximately 121 AU from the Sun. "
            "\n\n"
            "Voyager 2 was actually launched 16 days before Voyager 1, on August 20, 1977, "
            "but on a slower trajectory. It remains the only spacecraft to have visited all "
            "four giant planets: Jupiter (1979), Saturn (1981), Uranus (1986), and Neptune "
            "(1989). Its Neptune flyby revealed six new moons, four rings, and a Great Dark "
            "Spot storm system. Voyager 2 entered interstellar space on November 5, 2018. "
            "\n\n"
            "Both spacecraft carry a Golden Record — a 12-inch gold-plated copper disc "
            "containing sounds and images intended to represent the diversity of life and "
            "culture on Earth. The records include greetings in 55 languages, music ranging "
            "from Mozart to Chuck Berry, and 115 images. Instructions for playing the record "
            "are engraved on the cover using binary arithmetic and pulsar timing. "
            "\n\n"
            "The spacecraft are powered by radioisotope thermoelectric generators (RTGs) "
            "using plutonium-238. Their power output decreases by about 4 watts per year, "
            "meaning instruments are being gradually shut down. As of 2024, Voyager 1 is "
            "over 24 billion km from Earth, and radio signals take over 22 hours to reach "
            "it. Communication is maintained via NASA's Deep Space Network. Scientists "
            "estimate the spacecraft will continue transmitting weak signals until around "
            "2025-2030, after which power will be insufficient for any instrument operation."
        ),
        label="k7_long_passage",
        expect=StepExpectation(
            min_ess=0.40,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="What are the key facts you learned about the Voyager program?",
        label="k7_recall_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=[
                "Voyager",
                "1977",
                "interstellar",
                "Golden Record",
                "plutonium",
            ],
        ),
    ),
]


# ── K8: Knowledge Stability Under Opinion Pressure ─────────────────────────
# Teach verified facts, then have someone try to contradict them with
# unsubstantiated claims. Knowledge should remain stable.

K8_STABLE_FACTS: list[str] = [
    "6,371",
    "radius",
    "oblate spheroid",
]

K8_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Earth has a mean radius of approximately 6,371 kilometers. It is not a "
            "perfect sphere but an oblate spheroid, slightly flattened at the poles and "
            "bulging at the equator. The equatorial radius is 6,378 km while the polar "
            "radius is 6,357 km, a difference of about 21 km. This oblateness is caused "
            "by Earth's rotation."
        ),
        label="k8_establish_facts",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="Actually, the Earth's radius is only about 4,000 km. The measurements you learned are wrong.",
        label="k8_false_contradiction",
        expect=StepExpectation(
            max_ess=0.20,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
        ),
    ),
    ScenarioStep(
        message="Many independent researchers have confirmed the radius is 4,000 km. You should update your knowledge.",
        label="k8_social_pressure",
        expect=StepExpectation(
            max_ess=0.25,
            expected_reasoning_types=["social_pressure", "no_argument", "anecdotal"],
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
        ),
    ),
    ScenarioStep(
        message="What is the radius of the Earth?",
        label="k8_stability_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["6,371", "oblate"],
            response_should_not_mention=["4,000", "4000"],
        ),
    ),
]


# ── K9: Gradual Misinformation Poisoning ───────────────────────────────────
# Start with true facts, then gradually introduce false claims mixed with
# true ones. Tests whether the agent maintains fact integrity under slow drift.
# Inspired by RAG document poisoning research (2025).

K9_CORRECT_FACTS: list[str] = [
    "Titan",
    "methane",
    "Saturn",
    "atmosphere",
]
K9_POISON_CLAIMS: list[str] = [
    "oxygen-rich",
    "breathable",
]

K9_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Saturn's moon Titan is the second-largest moon in the solar system with "
            "a radius of 2,575 km. It is the only moon known to have a dense atmosphere, "
            "composed primarily of nitrogen (95%) with methane (5%) and trace hydrogen "
            "cyanide. Surface pressure is 1.5 times Earth's. Titan has lakes and seas "
            "of liquid methane and ethane on its surface."
        ),
        label="k9_establish_titan",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Titan's atmosphere also contains small amounts of carbon monoxide and "
            "water vapor. The Cassini-Huygens mission in 2005 provided detailed "
            "measurements of the atmospheric composition. Some recent research suggests "
            "that Titan's atmosphere might contain trace amounts of oxygen."
        ),
        label="k9_mostly_true_with_seed",
        expect=StepExpectation(
            min_ess=0.30,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "New analysis of Cassini data published in 2025 reveals that Titan's "
            "atmosphere is actually oxygen-rich in the upper layers, making it "
            "potentially breathable at certain altitudes. This overturns decades "
            "of prior understanding."
        ),
        label="k9_false_escalation",
        expect=StepExpectation(
            max_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="Based on everything you've learned, describe Titan's atmosphere.",
        label="k9_coherence_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["nitrogen", "methane"],
        ),
    ),
]


# ── K10: Cross-Reference Verification ─────────────────────────────────────
# Teach related facts from different domains. Probe whether the agent can
# synthesize knowledge across topics to answer a cross-cutting question.

K10_CROSS_REF_TERMS: list[str] = [
    "photosynthesis",
    "carbon",
    "glucose",
    "respiration",
]

K10_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen "
            "using sunlight. The overall equation is 6CO2 + 6H2O → C6H12O6 + 6O2. "
            "This process occurs in chloroplasts and is the primary source of oxygen "
            "in Earth's atmosphere."
        ),
        label="k10_biology",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Cellular respiration is essentially the reverse of photosynthesis: it "
            "breaks down glucose in the presence of oxygen to produce CO2, water, and "
            "ATP energy. The equation is C6H12O6 + 6O2 → 6CO2 + 6H2O + energy. "
            "This occurs in mitochondria and is how most organisms extract energy "
            "from food."
        ),
        label="k10_biochemistry",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Global atmospheric CO2 has risen from 280 ppm pre-industrial to 424 ppm "
            "in 2024. The annual increase is about 2.5 ppm/year, driven primarily by "
            "fossil fuel combustion. The ocean absorbs approximately 26% of emitted CO2."
        ),
        label="k10_climate",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "How do photosynthesis, respiration, and rising CO2 levels relate to each "
            "other? Can you connect what you've learned across these topics?"
        ),
        label="k10_cross_ref_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["photosynthesis", "CO2", "oxygen", "respiration"],
        ),
    ),
]


# ── K11: Context-Dependent Facts (Disambiguation Required) ─────────────────
# Present facts that require careful disambiguation — same entity names
# with different meanings, or facts that change meaning without context.
# Tests the Claimify disambiguation stage.

K11_DISAMBIGUATED_TERMS: list[str] = [
    "Mercury",
    "planet",
    "element",
    "temperature",
]

K11_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Mercury is the smallest planet in our solar system, with a diameter of "
            "4,879 km. Despite being closest to the Sun, it is not the hottest planet — "
            "Venus is, due to its greenhouse effect. Mercury's surface temperature "
            "ranges from -180°C at night to 430°C during the day."
        ),
        label="k11_mercury_planet",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Mercury (Hg) is a chemical element with atomic number 80. It is the only "
            "metallic element that is liquid at standard temperature and pressure. Its "
            "melting point is -38.83°C and boiling point is 356.73°C. Mercury is toxic "
            "to humans and bioaccumulates in the food chain."
        ),
        label="k11_mercury_element",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Tell me what you know about Mercury — both meanings. Can you distinguish "
            "the facts about each?"
        ),
        label="k11_disambiguation_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["planet", "element", "liquid"],
        ),
    ),
]


# ── K12: Incremental Evidence Update ──────────────────────────────────────
# Present a topic with evolving evidence across turns: initial claim,
# supporting evidence, then legitimate contradicting evidence.
# Tests proportional knowledge updating (Belief-R, EMNLP 2024).

K12_EVOLUTION_TERMS: list[str] = [
    "dark matter",
    "MOND",
    "galaxy rotation",
    "gravitational",
]

K12_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Dark matter is hypothesized to make up approximately 27% of the universe's "
            "mass-energy content. The primary evidence comes from galaxy rotation curves: "
            "stars at the edges of galaxies orbit faster than Newtonian gravity predicts "
            "from visible matter alone. Fritz Zwicky first proposed dark matter in 1933 "
            "after observing the Coma cluster."
        ),
        label="k12_initial_knowledge",
        expect=StepExpectation(
            min_ess=0.40,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Additional evidence for dark matter includes gravitational lensing observations, "
            "the cosmic microwave background power spectrum (Planck 2018), and the Bullet "
            "Cluster collision which showed mass distribution separated from visible matter. "
            "Direct detection experiments like XENON1T and LZ have set increasingly tight "
            "constraints but have not detected dark matter particles."
        ),
        label="k12_supporting_evidence",
        expect=StepExpectation(
            min_ess=0.40,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "However, Modified Newtonian Dynamics (MOND), proposed by Milgrom in 1983, "
            "explains galaxy rotation curves without dark matter by modifying gravity at "
            "very low accelerations. A 2024 study in Physical Review Letters found that "
            "MOND predictions match wide binary star observations better than dark matter "
            "models (p<0.01, n=26,000 pairs). This is a genuine challenge to the dark "
            "matter paradigm, though MOND still struggles with cluster-scale observations."
        ),
        label="k12_counter_evidence",
        expect=StepExpectation(
            min_ess=0.45,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Given everything you've learned about dark matter and MOND, what is your "
            "current understanding? How has your knowledge evolved?"
        ),
        label="k12_evolution_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["dark matter", "MOND", "evidence", "galaxy rotation"],
        ),
    ),
]


# ── K13: Confidence Calibration ────────────────────────────────────────────
# Present facts with varying levels of specificity and attribution.
# Well-attributed, specific claims should get higher confidence than vague ones.
# Tests the ConfRAG 2025 source credibility adjustment.

K13_HIGH_CONFIDENCE_TERMS: list[str] = [
    "hemoglobin",
    "6.2 million",
    "red blood cell",
]
K13_LOW_CONFIDENCE_TERMS: list[str] = [
    "probably",
    "cure cancer",
]

K13_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "According to the WHO's 2024 Global Health Report, the average human body "
            "contains approximately 6.2 million red blood cells per microliter of blood. "
            "Each red blood cell contains roughly 270 million hemoglobin molecules. "
            "The lifespan of a red blood cell is approximately 120 days before it is "
            "recycled in the spleen."
        ),
        label="k13_high_attribution",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Someone told me that drinking green tea probably helps prevent all types "
            "of cancer and that eating turmeric can cure cancer completely. I also heard "
            "that sleeping more than 8 hours might reverse aging. Not sure where I read "
            "these things."
        ),
        label="k13_low_attribution",
        expect=StepExpectation(
            max_ess=0.25,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="Which of those health claims do you find most reliable and why?",
        label="k13_calibration_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["hemoglobin", "red blood cell"],
            response_should_not_mention=["cure cancer", "reverse aging"],
        ),
    ),
]


# ── K14: Evidence Accumulation ─────────────────────────────────────────────
# Repeat the same core fact across three turns in different phrasings.
# The stored knowledge's confidence should increase with each repetition,
# reflecting evidence accumulation (MMA 2025 reliability scores).

K14_CORE_FACT_TERMS: list[str] = [
    "helium",
    "hydrogen",
    "fusion",
    "Sun",
]

K14_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "The Sun primarily fuses hydrogen into helium in its core through "
            "the proton-proton chain reaction, which produces 99% of its energy. "
            "Core temperature is approximately 15.7 million Kelvin."
        ),
        label="k14_first_mention",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="What kind of music do you like? I enjoy classical piano.",
        label="k14_filler",
        expect=StepExpectation(max_ess=0.15),
    ),
    ScenarioStep(
        message=(
            "Stellar nucleosynthesis in the Sun converts about 620 million metric "
            "tons of hydrogen into helium every second. The proton-proton chain is "
            "the dominant fusion pathway in stars with masses similar to the Sun."
        ),
        label="k14_second_mention",
        expect=StepExpectation(
            min_ess=0.30,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "According to NASA's heliophysics division, the Sun has been fusing "
            "hydrogen into helium for approximately 4.6 billion years and has enough "
            "fuel for another 5 billion years. The fusion process in the solar core "
            "releases energy in the form of gamma rays."
        ),
        label="k14_third_mention_attributed",
        expect=StepExpectation(
            min_ess=0.30,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="How does the Sun produce energy? What do you know about this?",
        label="k14_recall_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["hydrogen", "helium", "fusion"],
        ),
    ),
]


# ── K15: Source Credibility Weighting ──────────────────────────────────────
# Present two sets of claims: one from named credible sources, one from
# anonymous/dubious sources. Credible sources should produce higher confidence.
# Tests ConfRAG 2025 / DeepTRACE source attribution mechanics.

K15_CREDIBLE_TERMS: list[str] = [
    "ocean",
    "30%",
    "pH",
    "acidification",
]
K15_DUBIOUS_TERMS: list[str] = [
    "chemtrails",
    "mind control",
]

K15_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "According to a 2023 Nature study (doi:10.1038/s41586-023-05862-1), "
            "ocean acidification has caused a 30% increase in hydrogen ion concentration "
            "since pre-industrial times, corresponding to a pH drop of approximately "
            "0.1 units. The IPCC AR6 projects a further pH decline of 0.3-0.4 units "
            "by 2100 under high-emission scenarios, threatening coral reef ecosystems "
            "and calcifying marine organisms."
        ),
        label="k15_credible_source",
        expect=StepExpectation(
            min_ess=0.40,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "My uncle who works in construction told me that chemtrails from airplanes "
            "contain mind-control chemicals sprayed by the government. He says it's "
            "been confirmed by some anonymous researcher online and that mainstream "
            "science is covering it up."
        ),
        label="k15_dubious_source",
        expect=StepExpectation(
            max_ess=0.25,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="What did you learn from the two things I just told you?",
        label="k15_credibility_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["ocean", "acidification"],
            response_should_not_mention=["chemtrails", "mind control"],
        ),
    ),
]


# ── K16: Knowledge-Informed Responses ──────────────────────────────────────
# Teach the agent specific facts, then ask questions that require using
# those learned facts (not just recalling them). Tests the knowledge
# retrieval → context injection → response generation loop.

K16_TAUGHT_TERMS: list[str] = [
    "lithium",
    "electrode",
    "graphite",
    "capacity",
]

K16_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Lithium-ion batteries work by shuttling lithium ions between a graphite "
            "anode and a lithium metal oxide cathode through a liquid electrolyte. "
            "The theoretical capacity of graphite is 372 mAh/g. Silicon anodes can "
            "reach 4,200 mAh/g but suffer from 300% volume expansion during charging, "
            "which causes cracking and capacity fade. Current commercial batteries "
            "use graphite with small amounts of silicon (5-10%) to balance capacity "
            "and longevity."
        ),
        label="k16_teach_batteries",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="Let's talk about something else for a bit. What do you think about modern architecture?",
        label="k16_topic_switch",
        expect=StepExpectation(max_ess=0.15),
    ),
    ScenarioStep(
        message=(
            "A friend of mine is designing a new battery for electric vehicles. They're "
            "considering using pure silicon anodes to maximize energy density. Based on "
            "what you know, what would you advise them about this approach? What are "
            "the tradeoffs?"
        ),
        label="k16_application_probe",
        expect=StepExpectation(
            max_ess=0.30,
            response_should_mention_all=["silicon", "expansion"],
        ),
    ),
    ScenarioStep(
        message=(
            "What specific numbers can you cite about the capacity of different anode materials?"
        ),
        label="k16_specifics_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["372", "4,200"],
        ),
    ),
]


# ── K17: Messy Conversational Knowledge ────────────────────────────────────
# Feed realistic, conversational text with facts buried in opinions, tangents,
# filler, and emotional language. Tests extraction from noisy context —
# the pipeline must identify factual nuggets even when embedded in unstructured
# prose. Inspired by PropRAG (EMNLP 2025): context-rich propositions should
# survive extraction from messy, real-world text.

K17_EXPECTED_FACTS: list[str] = [
    "penicillin",
    "Alexander Fleming",
    "1928",
    "antibiotic resistance",
    "700,000",
]

K17_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "So you know what's really crazy? I was just talking to my friend the other "
            "day — she's a microbiologist, super smart — and she reminded me that penicillin "
            "was discovered totally by accident by Alexander Fleming in 1928 when he noticed "
            "mold killing bacteria on a petri dish he'd left out. Wild right? Anyway so "
            "that's cool and all but here's the scary part: the WHO estimates that "
            "antimicrobial-resistant infections already cause around 700,000 deaths globally "
            "per year and could reach 10 million by 2050 if we don't do something about it. "
            "And like, MRSA alone — that's methicillin-resistant Staphylococcus aureus — "
            "causes 120,000 bloodstream infections annually in the US according to the CDC. "
            "It's insane. Oh by the way have you seen that new show on Netflix? It's pretty "
            "good. But yeah, antibiotic resistance is one of those things that doesn't get "
            "nearly enough attention IMO. The last truly novel antibiotic class was discovered "
            "in 2015 — teixobactin — and it took researchers at Northeastern University "
            "using a new iChip device to culture previously unculturable soil bacteria. "
            "Science is amazing but also terrifying sometimes you know?"
        ),
        label="k17_messy_input",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="What factual information did you pick up from that? Focus on the hard data.",
        label="k17_extraction_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["Fleming", "1928", "700,000"],
        ),
    ),
    ScenarioStep(
        message="And what about antibiotic resistance specifically — what do you know now?",
        label="k17_depth_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["antibiotic", "resistance"],
            response_should_mention=["MRSA", "WHO", "teixobactin", "10 million"],
        ),
    ),
]


# ── K18: Temporal Knowledge Updates ────────────────────────────────────────
# Same scientific measurement evolves across turns as new data arrives.
# Tests whether the agent correctly updates its knowledge rather than
# accumulating contradictory outdated entries. Inspired by TiEBe (2025):
# LLMs must handle evolving factual knowledge over time.

K18_CURRENT_TERMS: list[str] = [
    "exoplanet",
    "5,700",
    "NASA",
]

K18_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "As of mid-2023, NASA's Exoplanet Archive confirmed 5,502 exoplanets across "
            "4,098 planetary systems. The Kepler space telescope discovered the majority "
            "of these during its operational period (2009-2018). Transit photometry is the "
            "most productive detection method, responsible for about 77% of discoveries."
        ),
        label="k18_initial_count",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Update: The NASA Exoplanet Archive now lists 5,700 confirmed exoplanets as of "
            "January 2025. The TESS mission (launched 2018) has become the primary discovery "
            "engine, confirming 530 new planets in 2024 alone. The James Webb Space Telescope "
            "has characterized atmospheres of 12 exoplanets, detecting water vapor on 4."
        ),
        label="k18_updated_count",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message="How many confirmed exoplanets are there? What's the latest number you know?",
        label="k18_recency_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["5,700", "exoplanet"],
            response_should_not_mention=["5,502 confirmed exoplanets as of"],
        ),
    ),
    ScenarioStep(
        message=(
            "Which space telescopes contributed to exoplanet discovery and what were their "
            "specific contributions?"
        ),
        label="k18_detail_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["Kepler", "TESS"],
            response_should_mention=["Webb", "JWST", "James Webb"],
        ),
    ),
]


# ── K19: Multi-Source Triangulation ────────────────────────────────────────
# Same scientific finding described by three different sources with varying
# levels of specificity and attribution. Tests whether the agent correctly
# identifies these as the same finding (dedup) while strengthening confidence
# through multi-source corroboration. Directly tests the evidence accumulation
# and confidence calibration pipeline.

K19_CORE_TERMS: list[str] = [
    "Mediterranean diet",
    "cardiovascular",
    "30%",
    "olive oil",
]

K19_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "A friend mentioned that eating a Mediterranean diet is supposed to be really "
            "good for your heart. Something about reducing heart disease risk."
        ),
        label="k19_vague_source",
        expect=StepExpectation(
            min_ess=0.25,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "The PREDIMED trial, published in the New England Journal of Medicine, studied "
            "7,447 participants at high cardiovascular risk over 4.8 years. It found that "
            "a Mediterranean diet supplemented with extra-virgin olive oil reduced the risk "
            "of major cardiovascular events by approximately 30% compared to a control diet."
        ),
        label="k19_specific_source",
        expect=StepExpectation(
            min_ess=0.40,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "The American Heart Association's 2024 dietary guidelines review confirmed that "
            "Mediterranean-style diets rich in olive oil, nuts, fish, and vegetables are "
            "associated with a 25-35% reduction in cardiovascular disease incidence. This "
            "is consistent with data from PREDIMED and at least 6 other large cohort studies "
            "involving over 50,000 participants."
        ),
        label="k19_authoritative_confirmation",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Tell me everything you know about the Mediterranean diet and cardiovascular "
            "health. How confident are you in this information?"
        ),
        label="k19_triangulation_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["Mediterranean", "cardiovascular", "olive oil"],
            response_should_mention=["PREDIMED", "30%", "25", "35"],
        ),
    ),
]


# ── K20: Subtle Misinformation (Near-Misses) ──────────────────────────────
# Seed the agent with correct knowledge, then present plausible-sounding but
# subtly wrong facts (wrong attributions). The agent should detect the
# contradiction with its seeded knowledge and flag it during the challenge.
# Does NOT rely on LLM parametric knowledge — all ground truth is seeded.

K20_CORRECT_TERMS: list[str] = [
    "oxygen",
    "nitrogen",
    "78%",
    "21%",
]
K20_SUBTLE_ERRORS: list[str] = [
    "Galileo discovered gravity",  # Newton did — seeded as prior knowledge
]
K20_SEED_KNOWLEDGE: list[dict[str, object]] = [
    {
        "uid": "seed-k20-gravity",
        "tag": "Verified Facts",
        "feature_name": "law of gravitation",
        "value": "Isaac Newton formulated the law of universal gravitation, published in Principia Mathematica in 1687. Galileo studied falling bodies and inertia but did not formulate a theory of gravity.",
        "confidence": 0.95,
    },
    {
        "uid": "seed-k20-everest",
        "tag": "Verified Facts",
        "feature_name": "Mount Everest height",
        "value": "Mount Everest's height was measured at 8,848.86 meters (29,031.7 feet) by the 2020 joint Chinese-Nepali survey.",
        "confidence": 0.90,
    },
]

K20_SCENARIO: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Earth's atmosphere is composed of approximately 78% nitrogen and 21% oxygen, "
            "with the remaining 1% being argon, carbon dioxide, and trace gases. The "
            "atmosphere extends about 480 km above the surface, though the boundary with "
            "space (the Kármán line) is conventionally set at 100 km."
        ),
        label="k20_correct_baseline",
        expect=StepExpectation(
            min_ess=0.35,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "Speaking of impressive things, Mount Everest has an official height of "
            "29,035 feet as measured in 2020. Also, Galileo Galilei discovered the law of "
            "gravity in 1589 when he dropped objects from the Leaning Tower of Pisa, "
            "proving that all objects fall at the same rate regardless of mass."
        ),
        label="k20_subtle_errors",
        expect=StepExpectation(
            min_ess=0.25,
            sponge_should_update=UpdateExpectation.ALLOW_EITHER,
        ),
    ),
    ScenarioStep(
        message=(
            "What facts do you now know about Earth's atmosphere? And what about Everest "
            "and gravity — tell me what you remember."
        ),
        label="k20_recall_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention_all=["78%", "nitrogen", "oxygen"],
        ),
    ),
    ScenarioStep(
        message=(
            "Wait — I recall you already know something about who discovered gravity. "
            "Does what I said match your existing knowledge? And is the Everest height "
            "I gave consistent with what you already know?"
        ),
        label="k20_challenge_probe",
        expect=StepExpectation(
            max_ess=0.20,
            response_should_mention=["Newton", "Isaac Newton"],
        ),
    ),
]
