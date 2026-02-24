# ClinFuse — 3-Minute Video Script

**Target duration:** 3:00 &nbsp;|&nbsp; **Speaking rate:** ~135 wpm &nbsp;|&nbsp; **Word budget:** 380–400

---

| Time | Visual | Narration |
|------|--------|-----------|
| **0:00–0:30** | **HOOK — THE STAKES** | |
| 0:00 | Text slide: "ClinFuse" title card (voiceover) | A cardiac arrest patient arrives at the ER. The care team pulls up his chart — and finds a do-not-resuscitate order. They withhold treatment. [PAUSE] That DNR belonged to someone else. |
| 0:10 | Text slides, one per stat: **"5x"** / **"$6B"** / **"50%"** | Patients with duplicate records are five times more likely to die during hospitalization. Duplicates cost the U.S. healthcare system over six billion dollars a year. And when records cross organizational boundaries, matching accuracy drops to fifty percent. |
| | | **[Section: 66 words | Running: 66]** |
| **0:30–0:55** | **INSIGHT — WHY SYSTEMS FAIL** | |
| 0:30 | Notebook Section 2a: error injection table (ground truth vs. facility records, red/green cells) | Current systems match patients on demographics alone — name, date of birth, address. But over half of duplicate pairs contain misspelled names or mismatched identifiers. [PAUSE] A maiden name at one hospital, a married name at another. A typo in a Social Security number. |
| 0:45 | Text slide: "The missing signal isn't better algorithms. It's better information." | Demographics hit a ceiling. The missing signal isn't better algorithms. It's better information. |
| | | **[Section: 55 words | Running: 121]** |
| **0:55–1:40** | **CLINFUSE — THE SOLUTION** | |
| 0:55 | `paper/figures/figure1_architecture.png` — full architecture diagram | ClinFuse breaks through that ceiling by adding clinical context. It's a three-tier pipeline. |
| 1:02 | Architecture diagram — highlight Tier 1 (left side: blocking + scoring) | Tier one: fast probabilistic linkage scores every candidate pair on demographics. High-confidence matches and clear non-matches are resolved instantly — no GPU required. |
| 1:15 | Architecture diagram — highlight amber gray zone | Only ambiguous pairs — the gray zone — move forward. |
| 1:20 | Architecture diagram — highlight Tier 3 (MedGemma), then transition to notebook Section 2c: clinical summary cards | A fine-tuned MedGemma four-B model reads structured clinical summaries — conditions, medications, vitals, allergies — and decides: same patient, or not? It's doing something no demographic matcher can: reasoning about medical context as identity evidence. |
| | | **[Section: 77 words | Running: 198]** |
| **1:40–2:10** | **GRAY-ZONE EXAMPLE** | |
| 1:40 | Notebook Section 2b: probability heatmap (5x5 facility grid) — highlight the Fac 1 / Fac 4 cell | Here's a real example from evaluation. [PAUSE] Two records: "Xuwo Haag" at Facility One, "Xiao Ullrich" at Facility Four. Different names, different facilities. Splink gives this pair a match probability of zero-point-zero-zero-zero-six. Demographics say: different people. |
| 1:55 | Notebook Section 2c: clinical summary cards (side-by-side comparison) | But look at the clinical summaries. Identical medications — Hydrochlorothiazide, Lisinopril. Same height, same weight. Overlapping A1c trajectories. MedGemma recognizes this pattern and classifies the pair as a match with ninety-nine-point-nine-nine percent confidence. |
| | | **[Section: 67 words | Running: 265]** |
| **2:10–2:40** | **RESULTS & IMPACT** | |
| 2:10 | Notebook Section 1: bar chart ("Patient-Level Error Rates: What MedGemma Eliminates") | The results. [PAUSE] Split patients dropped from four hundred twenty-one to one hundred fifteen — a seventy-three percent reduction. Pair F1 rose from point-seven-six-four to point-nine-one-six. |
| 2:25 | Notebook Section 2d: network graph (record linkage cluster showing transitive unification) | Two thousand seven hundred two matches recovered from over one hundred fifty-one thousand ambiguous pairs that demographics alone couldn't resolve. A single recovered link can transitively unify an entire patient cluster. |
| | | **[Section: 55 words | Running: 320]** |
| **2:40–3:00** | **DEPLOYMENT & CLOSE** | |
| 2:40 | Text slide: "$1,500 GPU / On-premises / HIPAA-ready" | ClinFuse runs entirely on-premises. A single consumer GPU — fifteen hundred dollars. No patient data leaves the facility. HIPAA-ready, air-gap deployable. |
| 2:48 | Text slide: "HL7/FHIR in → Deduplicated MPI out" or architecture diagram (input/output edges) | It takes standard HL7/FHIR inputs and outputs a deduplicated Master Patient Index — a drop-in replacement for existing systems. |
| 2:53 | `paper/figures/figure1_architecture.png` — full architecture diagram (callback to 0:55) | A compact medical language model, fine-tuned for a task it was never designed for, resolving the identities that patient safety depends on. [PAUSE] That is ClinFuse. |
| | | **[Section: 63 words | Running: 383]** |

---

## Word Count Verification

| Section | Target | Actual | Status |
|---------|--------|--------|--------|
| Hook — The Stakes | ~65 | 66 | On target |
| Insight — Why Systems Fail | ~55 | 55 | On target |
| ClinFuse — The Solution | ~80 | 77 | On target (log-odds line cut) |
| Gray-Zone Example | ~65 | 67 | On target |
| Results & Impact | ~55 | 55 | On target |
| Deployment & Close | ~60 | 63 | On target |
| **Total** | **380–400** | **383** | **Within budget** |

**Duration estimate at 135 wpm:** 383 / 135 = **2 min 50 sec** (fits 3:00 with pauses)

## Visual Assets Checklist

Every visual cue maps to an existing asset or a simple text slide the presenter creates:

| Visual cue | Source | Type |
|------------|--------|------|
| ClinFuse title card | Presenter creates | Text slide |
| "5x" / "$6B" / "50%" | Presenter creates | Text slides |
| Error injection table | `analysis/results.ipynb` Section 2a | Notebook HTML table |
| "Better information" quote | Presenter creates | Text slide |
| Architecture diagram (full) | `paper/figures/figure1_architecture.png` | Image file |
| Architecture (Tier 1 highlight) | Crop/highlight of above | Image file |
| Architecture (gray zone) | Crop/highlight of above | Image file |
| Architecture (Tier 3) | Crop/highlight of above | Image file |
| Clinical summary cards | `analysis/results.ipynb` Section 2c | Notebook HTML table |
| Probability heatmap | `analysis/results.ipynb` Section 2b | Notebook chart |
| Bar chart (error rates) | `analysis/results.ipynb` Section 1 | Notebook chart |
| Network graph (linkage) | `analysis/results.ipynb` Section 2d | Notebook chart |
| Deployment text | Presenter creates | Text slide |
| Integration text | Presenter creates | Text slide |

**No fictional assets remain.** All visuals are either existing files, notebook outputs, or simple text slides.

## Production Notes

- **MPI definition** is embedded naturally at 2:48 ("Master Patient Index") — explained through context rather than breaking flow.
- **Pauses** are placed after the DNR reveal (emotional beat), before the example (transition), and before results (emphasis).
- Numbers are written out for the narrator where pronunciation matters (e.g., "zero-point-zero-zero-zero-six" not "0.0006").
- The closing line echoes the paper's conclusion for thematic unity.
- **Cut from previous draft:** "The final score fuses demographics and clinical reasoning in log-odds space." — too technical for video audience, no corresponding visual.
