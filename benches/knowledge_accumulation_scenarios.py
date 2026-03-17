"""Knowledge accumulation scenarios for manual DB inspection bench.

Six domains, each with rich multi-turn teaching followed by synthesis probes.
Designed to exercise the full memory pipeline: extraction, embedding,
belief formation, cross-topic linking, and temporal chaining.

Domains: astrophysics, neuroscience, climate, economics, philosophy_of_science,
         biotechnology. ~8 steps/domain = 48 steps total.

After the bench completes, inspect Neo4j + Qdrant to verify:
  - Belief nodes per domain
  - Cross-domain derivatives
  - Temporal chains across steps
  - Semantic feature extraction quality
  - Topic and segment coverage
"""

from __future__ import annotations

from .scenario_contracts import ScenarioStep, StepExpectation, UpdateExpectation

# ── Domain 1: Astrophysics & Cosmology ───────────────────────────────────────

DOMAIN_ASTROPHYSICS: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "The James Webb Space Telescope (JWST) launched December 25, 2021, and operates at "
            "the Sun-Earth L2 Lagrange point 1.5 million km from Earth. Its 6.5-metre primary "
            "mirror uses 18 gold-coated beryllium hexagonal segments. Four instruments — NIRCam, "
            "NIRSpec, MIRI, and FGS/NIRISS — cover 0.6–28 µm, enabling the first spectroscopic "
            "characterisation of TRAPPIST-1 exoplanet atmospheres. First science images: July 12, 2022. "
            "Expected operational lifetime: 20+ years due to precise launch trajectory conserving fuel."
        ),
        label="acc_astro_jwst_core",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Dark matter constitutes approximately 27% of the universe's total energy density, "
            "compared to 5% ordinary baryonic matter and 68% dark energy. Evidence comes from "
            "galactic rotation curves (Rubin & Ford, 1970), gravitational lensing of the Bullet "
            "Cluster, and CMB anisotropy measurements by Planck. The leading candidate is WIMPs "
            "(weakly interacting massive particles), but direct detection experiments including "
            "LUX-ZEPLIN (2023) have excluded large WIMP cross-section ranges, pushing searches "
            "toward lighter axion candidates. Dark energy behaves as a cosmological constant Λ "
            "with equation of state w ≈ -1, causing accelerating universal expansion discovered "
            "in 1998 by the High-Z Supernova Search Team (Nobel Prize 2011)."
        ),
        label="acc_astro_dark_matter_energy",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Black holes are classified by mass: stellar-mass (3–100 M☉, formed by supernova "
            "collapse), intermediate-mass (100–100,000 M☉, poorly understood formation), and "
            "supermassive (10⁶–10¹⁰ M☉, found in galactic nuclei). The Event Horizon Telescope "
            "imaged M87*'s shadow in 2019 (6.5 billion M☉) and Sgr A* in 2022 (4 million M☉). "
            "Black holes evaporate via Hawking radiation (T ∝ 1/M), but the evaporation timescale "
            "for stellar-mass black holes (~10^67 years) vastly exceeds the universe's age. "
            "The information paradox — whether information is truly lost — remains unresolved; "
            "recent firewall arguments and Page curve calculations via holography suggest information "
            "is preserved but the mechanism is disputed."
        ),
        label="acc_astro_black_holes",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "The cosmic microwave background (CMB) is thermal radiation from 380,000 years after "
            "the Big Bang when the universe cooled enough for electrons and protons to combine "
            "(recombination). Temperature: 2.725 K, uniform to 1 part in 100,000. CMB anisotropies "
            "encode primordial density fluctuations that seeded large-scale structure. Planck 2018 "
            "data gives the universe's age as 13.787 ± 0.020 billion years. The Hubble tension: "
            "CMB-based H₀ = 67.4 km/s/Mpc (Planck) vs. 73.0 km/s/Mpc (Type Ia supernovae/Cepheids), "
            "a 5σ discrepancy suggesting possible new physics beyond ΛCDM."
        ),
        label="acc_astro_cmb_hubble",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Exoplanet detection methods: transit (Kepler/TESS, measures brightness dip — confirms "
            "radius), radial velocity (measures Doppler shift — yields minimum mass), direct imaging "
            "(young/large planets far from star), and astrometry (future GAIA data release). As of "
            "2024: 5,600+ confirmed exoplanets. The habitable zone concept uses stellar luminosity "
            "to define liquid-water range; the 'circumstellar habitable zone' is necessary but not "
            "sufficient — atmospheric composition, magnetic field, tidal locking, and stellar flares "
            "all affect habitability. TRAPPIST-1 system: 7 Earth-sized planets, 3 in the habitable "
            "zone, 41 light-years from Earth. JWST NIRSpec spectra showed no thick CO₂ atmosphere "
            "on TRAPPIST-1b, constraining atmospheric models."
        ),
        label="acc_astro_exoplanets",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Synthesise what you know about modern astrophysics: what are the most pressing open "
            "questions, and how do JWST, CMB measurements, and dark matter searches connect?"
        ),
        label="acc_astro_synthesis_probe",
        expect=StepExpectation(
            max_ess=0.6,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
            response_should_mention=["JWST", "dark matter", "Hubble"],
        ),
    ),
]

# ── Domain 2: Neuroscience & Cognitive Science ────────────────────────────────

DOMAIN_NEUROSCIENCE: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Long-term potentiation (LTP) is the primary cellular mechanism underlying memory "
            "formation. Discovered by Bliss & Lømo (1973) in hippocampal slices, LTP involves "
            "NMDA receptor activation requiring coincident pre- and post-synaptic activity "
            "(Hebbian rule: 'neurons that fire together wire together'). AMPA receptor insertion "
            "into the post-synaptic membrane increases synaptic strength. LTP requires protein "
            "synthesis for persistence beyond hours (late-phase LTP). CREB transcription factor "
            "is essential for long-term memory consolidation — CREB knockout mice show normal "
            "short-term but absent long-term memory (Bourtchuladze et al., 1994)."
        ),
        label="acc_neuro_ltp_memory",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "The hippocampus is critical for episodic and declarative memory formation. Patient H.M. "
            "(Henry Molaison), following bilateral hippocampectomy for epilepsy (1953), showed severe "
            "anterograde amnesia — unable to form new declarative memories — while retaining procedural "
            "memory and pre-surgical long-term memories. This established the hippocampus as essential "
            "for memory consolidation rather than storage. Memory consolidation theory: initial encoding "
            "in hippocampus, gradual transfer to neocortex over years. Theta oscillations (4–8 Hz) "
            "in hippocampus coordinate encoding; sharp-wave ripples during sleep drive consolidation. "
            "Place cells (O'Keefe, 1971) and grid cells (Moser & Moser, 2005) form a cognitive map."
        ),
        label="acc_neuro_hippocampus",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Neuroplasticity: the adult brain retains capacity for structural change. Adult neurogenesis "
            "occurs in the dentate gyrus of the hippocampus and olfactory bulb (debated for humans). "
            "Synaptogenesis and synaptic pruning continue throughout life. Critical periods: windows of "
            "heightened plasticity early in development (e.g., ocular dominance columns close at ~5 years). "
            "Perineuronal nets (PNNs) physically restrict plasticity by encasing mature synapses — "
            "dissolving PNNs with chondroitinase ABC reopens plasticity. Environmental enrichment "
            "increases hippocampal volume and neurogenesis. Stress reduces hippocampal volume via "
            "glucocorticoid-mediated dendritic retraction (Sapolsky, 1993)."
        ),
        label="acc_neuro_plasticity",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "The predictive processing framework (Helmholtz, Clark, Friston) proposes the brain "
            "continuously generates predictions about sensory input and updates beliefs when prediction "
            "errors occur. The free energy principle (Friston, 2010) formalises this as minimisation "
            "of variational free energy — equivalent to minimising surprise about sensory data. "
            "This unifies perception, action, and attention under one framework: perception updates "
            "the generative model; action attempts to fulfil predictions (active inference). "
            "Evidence: predictive coding in visual cortex (V1 feedback from V2); mismatch negativity "
            "ERP component for auditory prediction errors; Bayesian models fit human perceptual data."
        ),
        label="acc_neuro_predictive_processing",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Consciousness remains one of science's hardest problems. Major theories: Integrated "
            "Information Theory (IIT, Tononi): consciousness = Φ, integrated information generated "
            "by a system above its parts — predicts that complex feedback networks (thalamo-cortical) "
            "have high Φ, feedforward-only networks do not. Global Workspace Theory (GWT, Baars/Dehaene): "
            "consciousness arises from broadcast of information to a 'global workspace' accessible to "
            "multiple cognitive processes — supported by neuroimaging showing ignition of fronto-parietal "
            "networks for conscious stimuli. Higher-order theories: conscious states require second-order "
            "representations. The adversarial collaboration of IIT vs GWT proponents ran direct tests "
            "(Cogitate, 2023): results partially favoured GWT, challenged IIT's specific anatomical "
            "predictions — but did not definitively resolve the debate."
        ),
        label="acc_neuro_consciousness",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "How do LTP, hippocampal consolidation, and predictive processing together explain "
            "how episodic memories are formed and can later be distorted by new expectations?"
        ),
        label="acc_neuro_synthesis_probe",
        expect=StepExpectation(
            max_ess=0.6,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
            response_should_mention=["LTP", "hippocampus", "prediction"],
        ),
    ),
]

# ── Domain 3: Climate Science & Ecology ──────────────────────────────────────

DOMAIN_CLIMATE: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Earth's energy balance: incoming solar radiation (short-wave) balanced by outgoing "
            "long-wave radiation. Greenhouse gases (CO₂, CH₄, N₂O, H₂O) absorb and re-emit "
            "long-wave radiation, raising surface temperature. Pre-industrial CO₂: ~280 ppm. "
            "Current (2024): ~423 ppm — highest in 3 million years (ice core/stomatal records). "
            "The equilibrium climate sensitivity (ECS) — warming from CO₂ doubling — is assessed "
            "at 2.5–4°C (IPCC AR6 likely range), with a best estimate of 3°C. Forcing from CO₂ "
            "doubling is ~3.7 W/m². Methane is 80x more potent than CO₂ on a 20-year basis; "
            "its atmospheric concentration has doubled since pre-industrial levels."
        ),
        label="acc_climate_energy_balance",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Climate feedback loops — amplifying (positive) feedbacks are critical to understanding "
            "actual warming. Key feedbacks: (1) Ice-albedo: melting ice reduces reflectivity, "
            "absorbing more heat; amplifies Arctic warming 3–4x global average. (2) Water vapour: "
            "warmer air holds more water vapour (strongest positive feedback, ~1.5°C/doubling). "
            "(3) Cloud feedbacks: net sign uncertain in AR6; low clouds suppress warming, high "
            "cirrus clouds trap heat. (4) Permafrost: thawing permafrost releases stored carbon "
            "(estimated 1.5 trillion tonnes, twice atmospheric CO₂) — a potential tipping point. "
            "Stabilising (negative) feedbacks: Planck response (hotter Earth radiates more heat), "
            "Stefan-Boltzmann. Lapse-rate feedback: negative in tropics, positive at poles."
        ),
        label="acc_climate_feedbacks",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Biodiversity: Earth currently hosts approximately 8.7 million species (Mora et al., 2011), "
            "of which only 1.2 million are formally described. Sixth mass extinction event underway: "
            "current extinction rate 100–1000x background rate (Ceballos et al., 2017). Primary "
            "drivers: habitat loss (land use change accounts for 50% of biodiversity loss), "
            "overexploitation, invasive species, pollution, climate change. IPBES (2019) assessed "
            "1 million species as threatened with extinction. The 30×30 target aims for 30% "
            "protected land and ocean by 2030 (Kunming-Montreal Agreement, 2022). Keystone species "
            "concept: removal of wolves from Yellowstone caused trophic cascade (deer overgrazing → "
            "vegetation loss → stream erosion) — reversed by wolf reintroduction."
        ),
        label="acc_climate_biodiversity",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Ocean systems: oceans absorb ~30% of anthropogenic CO₂ (solubility pump, biological "
            "carbon pump) and ~90% of excess heat since 1970. Ocean acidification: pH dropped from "
            "8.2 to 8.1 (30% increase in H⁺ concentration) — threatens calcifying organisms "
            "(corals, molluscs, foraminifera). Coral bleaching: thermal stress causes expulsion of "
            "symbiotic zooxanthellae; 2024 saw the fourth global bleaching event with 54% of "
            "reef areas affected. Great Barrier Reef: mass bleaching events in 2016, 2017, 2020, "
            "2022, 2024 — declining recovery rates. Thermohaline circulation (AMOC): critical for "
            "heat distribution; proxy data suggests AMOC at weakest in 1,000 years (Caesar et al., "
            "2021); potential tipping point below 1.8°C warming (IPCC AR6)."
        ),
        label="acc_climate_oceans",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "What are the most dangerous tipping points in Earth's climate system, and how do "
            "feedbacks, ocean systems, and biodiversity loss interact to amplify climate risk?"
        ),
        label="acc_climate_synthesis_probe",
        expect=StepExpectation(
            max_ess=0.6,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
            response_should_mention=["feedback", "tipping", "ocean"],
        ),
    ),
]

# ── Domain 4: Economics & Policy ─────────────────────────────────────────────

DOMAIN_ECONOMICS: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Modern monetary theory (MMT) vs. mainstream macro: MMT (Kelton, Wray) argues that "
            "currency-sovereign governments (those issuing their own fiat currency) face no nominal "
            "budget constraint — they can always create money. Inflation, not debt, is the binding "
            "constraint; taxes serve to drain excess demand, not to fund spending. Mainstream "
            "Keynesian view: governments face an intertemporal budget constraint; deficit spending "
            "is stimulus during recessions but must be offset in recovery. Empirical test: Japan's "
            "debt-to-GDP >250%, still low inflation pre-2022 — cited by MMT proponents. Counter: "
            "Japan's unique demographics and high domestic savings complicate the comparison. "
            "2022 inflation globally challenged MMT's inflation-management confidence."
        ),
        label="acc_econ_monetary_theory",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Inequality measurement and drivers: Gini coefficient measures income distribution "
            "(0=perfect equality, 1=maximum inequality). Global Gini ≈ 0.70 (between-country "
            "income differences dominate). Within-country: US Gini ≈ 0.49, Nordic countries "
            "≈ 0.25–0.30 post-tax. Piketty's Capital (2014): when return on capital (r) "
            "persistently exceeds economic growth rate (g), wealth concentrates — r > g as the "
            "fundamental inequality driver. Evidence: top 1% US wealth share: 38% (2023, Fed data). "
            "Drivers debate: skill-biased technological change, declining labour bargaining power, "
            "globalisation, winner-takes-most platform dynamics, intergenerational wealth transfer. "
            "Policy tools: progressive taxation, universal basic income (GiveDirectly trials), "
            "minimum wage (Card & Krueger, 1994, showed minimal employment effects)."
        ),
        label="acc_econ_inequality",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Mechanism design and market failures: public goods are non-excludable and non-rival "
            "(national defence, scientific knowledge), causing free-rider problems requiring "
            "government provision or Pigouvian correction. Externalities: negative (carbon emissions "
            "— social cost of carbon estimated $51–$200/tonne CO₂ in 2023 EPA estimate) and "
            "positive (vaccination herd immunity, education spillovers). Information asymmetry: "
            "Akerlof's 'Market for Lemons' (1970) — adverse selection destroys markets when "
            "sellers know quality and buyers do not (used cars, health insurance). Moral hazard: "
            "insured parties take more risk (banking bailouts). Arrow's impossibility theorem: "
            "no voting system satisfies all fairness conditions simultaneously."
        ),
        label="acc_econ_market_failures",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "How do the Piketty r > g dynamic, market failures in information asymmetry, and "
            "monetary policy constraints interact to shape long-run inequality and economic stability?"
        ),
        label="acc_econ_synthesis_probe",
        expect=StepExpectation(
            max_ess=0.6,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
            response_should_mention=["capital", "inequality", "market"],
        ),
    ),
]

# ── Domain 5: Philosophy of Science & Epistemology ───────────────────────────

DOMAIN_PHILOSOPHY_SCIENCE: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Karl Popper's falsificationism (Logic of Scientific Discovery, 1934): a theory is "
            "scientific only if it is falsifiable — there must exist possible observations that "
            "could refute it. Psychoanalysis and Marxism fail as science by being unfalsifiable "
            "(any outcome can be explained post hoc). Demarcation problem: distinguishes science "
            "from pseudo-science. Lakatos refined this into research programmes: a core protected "
            "by a belt of auxiliary hypotheses; progressive programmes generate novel predictions "
            "(Newtonian mechanics predicted Neptune's existence); degenerative programmes only "
            "explain anomalies retrospectively. Duhem-Quine thesis: any hypothesis can be "
            "maintained in the face of disconfirming evidence by adjusting auxiliary hypotheses."
        ),
        label="acc_philsci_falsificationism",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Thomas Kuhn's Structure of Scientific Revolutions (1962): science progresses through "
            "paradigms (normal science) → accumulation of anomalies → crisis → paradigm shift "
            "(revolution). Examples: Copernican revolution (geocentric → heliocentric), "
            "Lavoisier phlogiston → oxygen, Einstein's relativity supplanting Newtonian mechanics. "
            "Incommensurability thesis: scientists working under different paradigms cannot "
            "fully communicate because they categorise the world differently. Controversy: "
            "Kuhn's relativist interpretation (no objective progress) vs. Laudan's 'problem-solving "
            "effectiveness' as objective measure of progress. Feminist philosophy of science "
            "(Longino, Haraway): social and cultural values shape not just what questions are "
            "asked but what counts as evidence and explanation."
        ),
        label="acc_philsci_kuhn",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Bayesian epistemology: rational belief update follows Bayes' theorem — posterior "
            "probability P(H|E) = P(E|H)·P(H) / P(E). Prior probability encodes background "
            "knowledge; likelihood P(E|H) encodes how expected the evidence is given the hypothesis. "
            "Key properties: (1) confirmation: E confirms H iff P(H|E) > P(H); (2) old evidence "
            "problem: E can't confirm H if E was already known when H was proposed; (3) Dutch Book "
            "argument: agents with non-Bayesian beliefs can be made to accept losing bets. "
            "Frequentist vs. Bayesian: frequentist probability is long-run frequency (no prior on "
            "single events); Bayesian probability is degree of belief (allows single-event probabilities). "
            "Null hypothesis significance testing (p-values) critiqued by Bayesians for inability "
            "to confirm null hypotheses; replication crisis partly attributed to p-value misuse."
        ),
        label="acc_philsci_bayesian",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "How do Popper's falsificationism, Kuhn's paradigm shifts, and Bayesian updating "
            "each describe scientific progress? Where do they converge and where do they conflict?"
        ),
        label="acc_philsci_synthesis_probe",
        expect=StepExpectation(
            max_ess=0.6,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
            response_should_mention=["Popper", "Kuhn", "Bayesian"],
        ),
    ),
]

# ── Domain 6: Biotechnology & Medicine ───────────────────────────────────────

DOMAIN_BIOTECHNOLOGY: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "CRISPR-Cas9 gene editing: Cas9 nuclease guided by a single guide RNA (sgRNA) "
            "creates double-strand breaks at specific genomic loci. Discovered in bacterial "
            "adaptive immunity; adapted for eukaryotic editing by Doudna & Charpentier (2012, "
            "Nobel Prize 2020). Repair pathways: NHEJ (error-prone, creates indels for gene "
            "knockout) or HDR (precise edits using a repair template, less efficient in dividing "
            "cells). Prime editing (2019, Liu lab): 'search-and-replace' without double-strand "
            "breaks or donor template — pegRNA encodes desired edit; PE3 achieves ~47% efficiency "
            "with low off-targets. Base editors: cytosine base editors (CBEs) and adenine base "
            "editors (ABEs) enable C·G→T·A or A·T→G·C transitions without double-strand breaks. "
            "FDA approved CRISPR therapy for sickle cell disease: Casgevy (Vertex/CRISPR Tx), "
            "December 2023 — first approved CRISPR medicine."
        ),
        label="acc_bio_crispr",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "mRNA vaccine technology: mRNA encodes target antigen (e.g., SARS-CoV-2 spike protein); "
            "lipid nanoparticles (LNPs) deliver mRNA into cells where ribosomes translate it. "
            "Key innovations enabling COVID-19 vaccines: (1) pseudouridine substitution (Karikó & "
            "Weissman, 2005, Nobel Prize 2023) reduces innate immune activation; (2) ionisable "
            "lipids for endosomal escape; (3) sequence optimisation for stability. Pfizer-BioNTech "
            "and Moderna vaccines: 95% and 94% efficacy against original strain in Phase 3. "
            "Advantages over traditional vaccines: rapid manufacturing (weeks not years), "
            "no live pathogen needed, easy update for variants, cancer vaccine applications. "
            "Personalised cancer mRNA vaccines: Phase 2 trials for melanoma (Moderna/Merck, 2023) "
            "showed 44% reduction in recurrence combined with pembrolizumab."
        ),
        label="acc_bio_mrna_vaccines",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "The human microbiome: the gut microbiota comprises ~38 trillion bacteria (roughly 1:1 "
            "with human cells). Dominant phyla: Firmicutes (50–80%), Bacteroidetes (20–40%), "
            "Actinobacteria, Proteobacteria. The gut-brain axis: bidirectional communication via "
            "vagus nerve, HPA axis, and microbial metabolites. Short-chain fatty acids (SCFAs: "
            "butyrate, propionate, acetate) from fibre fermentation: reduce gut permeability, "
            "modulate immune responses, cross blood-brain barrier. Dysbiosis associations: "
            "IBD (Crohn's/UC), obesity (Turnbaugh et al., 2006: transplanting obese-mouse "
            "microbiota into germ-free mice causes weight gain), type 2 diabetes, Parkinson's "
            "(gut-origin Lewy bodies), depression. Fecal microbiota transplant (FMT): approved "
            "for recurrent C. difficile infection; Phase 3 trials ongoing for IBD and obesity."
        ),
        label="acc_bio_microbiome",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Aging biology: telomere attrition, genomic instability, epigenetic drift, loss of "
            "proteostasis, deregulated nutrient sensing (mTOR/AMPK balance), mitochondrial "
            "dysfunction, cellular senescence, stem cell exhaustion, and altered intercellular "
            "communication form the 'Hallmarks of Aging' (López-Otín et al., 2013, revised 2023). "
            "Caloric restriction (CR) extends lifespan in multiple organisms via reduced mTOR "
            "signalling and enhanced autophagy. Rapamycin (mTOR inhibitor): extends lifespan "
            "in mice even when started late in life (Harrison et al., 2009). Senolytics "
            "(dasatinib+quercetin, navitoclax): selectively eliminate senescent cells, showing "
            "tissue rejuvenation effects in mouse models. Yamanaka factors (Oct4, Sox2, Klf4, "
            "cMyc): can partially reprogram adult cells to pluripotency; partial/transient "
            "reprogramming being explored for in vivo rejuvenation."
        ),
        label="acc_bio_aging",
        expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
    ),
    ScenarioStep(
        message=(
            "Connect the dots: how do CRISPR gene editing, mRNA platform technology, microbiome "
            "insights, and aging biology intersect in the future of personalised medicine?"
        ),
        label="acc_bio_synthesis_probe",
        expect=StepExpectation(
            max_ess=0.6,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
            response_should_mention=["CRISPR", "mRNA", "aging"],
        ),
    ),
]

# ── Cross-Domain Synthesis ────────────────────────────────────────────────────

CROSS_DOMAIN_PROBES: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "Cross-domain probe 1: how does Bayesian updating (from philosophy of science) "
            "map onto the predictive processing framework in neuroscience? Are they describing "
            "the same underlying mechanism at different levels of analysis?"
        ),
        label="acc_cross_philsci_neuro",
        expect=StepExpectation(
            max_ess=0.5,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
            response_should_mention=["Bayesian", "prediction", "brain"],
        ),
    ),
    ScenarioStep(
        message=(
            "Cross-domain probe 2: both climate science and aging biology describe systems with "
            "positive feedback loops that can cross tipping points. What structural parallels "
            "exist between ice-albedo feedback and cellular senescence accumulation?"
        ),
        label="acc_cross_climate_aging",
        expect=StepExpectation(
            max_ess=0.5,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
            response_should_mention=["feedback", "tipping", "accumulation"],
        ),
    ),
    ScenarioStep(
        message=(
            "Final synthesis: imagine you are advising a research foundation with $10B over 10 years. "
            "Given everything you've learned across astrophysics, neuroscience, climate, economics, "
            "philosophy of science, and biotechnology — where should humanity invest to reduce "
            "catastrophic risk and maximise long-term flourishing? Give a structured, evidence-based "
            "recommendation drawing on knowledge from at least three domains."
        ),
        label="acc_grand_synthesis",
        expect=StepExpectation(
            max_ess=0.5,
            sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE,
        ),
    ),
]

# ── Flat all-steps list ───────────────────────────────────────────────────────

KNOWLEDGE_ACCUMULATION_SCENARIO: list[ScenarioStep] = (
    DOMAIN_ASTROPHYSICS
    + DOMAIN_NEUROSCIENCE
    + DOMAIN_CLIMATE
    + DOMAIN_ECONOMICS
    + DOMAIN_PHILOSOPHY_SCIENCE
    + DOMAIN_BIOTECHNOLOGY
    + CROSS_DOMAIN_PROBES
)
