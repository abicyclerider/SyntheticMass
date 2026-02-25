# ClinFuse — 3-Minute Video Script

**Format:** Screen recording of `analysis/results.ipynb` with voiceover. All visuals are notebook cells — scroll straight down, never back up.

**Target duration:** 3:00 | **Speaking rate:** ~135 wpm | **Word budget:** ~380

---

## ACT 1 — The Problem (~45 sec)

*Visible: title + executive summary (cells 1–2). No scrolling yet.*

In 2016, a cardiac arrest patient arrived at an ER in the southern United States. The care team pulled up his chart, found a do-not-resuscitate order, and withheld treatment. [PAUSE] That DNR belonged to a different patient.

At a more macro level, mismatching patients to their medical records costs the U.S. healthcare system over six billion dollars a year.

To explore how MedGemma [med-JEM-uh] can be used to tackle this challenge, we simulated a fragmented healthcare system using Synthea and a custom augmentation pipeline. We then fed these simulated medical records through ClinFuse, a MedGemma enabled entity resolution pipeline to observe the performance.


*[scroll to architecture diagram]*

ClinFuse works in three tiers. First, a probabilistic linker compares surface-level fields like name, date of birth, and address to find candidate pairs. Then a fine-tuned MedGemma model reads deeper medical information — medications, lab results, diagnoses — to resolve the cases those surface fields can't decide. Finally, golden records of resolved patient identities are assembled in the third stage.

**[98 words]**

---

## ACT 2 — The Solution in Action (~90 sec)

*Visible: case study cells 3–19. Continuous scrolling, driven by narration.*

Lets take a look at ClinFuse in action.

Xiao Haag is a simulated forty-seven-year-old woman who visited five hospitals, each time registering as a new patient.

[scroll to error table]

Her records are full of data entry errors — typos, maiden names, missing fields.

[scroll to heatmap]

The best pairwise probability? Zero point zero zero zero six. Demographics say: different people.

[scroll to clinical summaries]

This is where MedGemma takes over. Look at what it sees. Identical medications — Hydrochlorothiazide [hy-droh-klor-oh-THY-uh-zide], Lisinopril. Same height, same weight. Overlapping A1c trajectories. The same chronic conditions managed across years. MedGemma says: same patient, ninety-nine point nine nine percent confidence.

[scroll to decisions table]

Nine of ten links recovered.

[scroll to network graph]

One recovered link transitively unifies the entire cluster — five facilities, one identity.

[scroll to golden record]

The golden record: the correct name, date of birth, and address — the complete identity recovered from fragments no single facility got right.

**[183 words | Running: 281]**

---

## ACT 3 — Results + Deployment (~45 sec)

*Visible: results cells 20–22, then summary cell 26.*

[scroll to bar chart]

We tested against twenty-five hundred synthetic patients spread across five facilities, with eight to twelve demographic errors injected per record — typos, maiden names, missing fields. Far worse than production EHR data.

[scroll to summary]

Four hundred twenty-one of those patients had fragmented records when information from medical records was not used to resolve the patient identities. Applying MedGemma brought that down to one hundred fifteen. [PAUSE] Seventy-three percent fewer patients at risk.

Because ClinFuse only sends ambiguous cases to MedGemma, the 4B parameter model can run inference on a modest consumer GPU, this technique scales to larger record populations and on-prem deployment is practical sidestepping many HIPAA and GDPR privacy concerns entirely.
**[88 words | Running: 369]**

---

| Act | Words | Content |
|-----|-------|---------|
| Act 1: The Problem | ~98 | DNR story + cost + user bridge + ClinFuse reveal |
| Act 2: Solution in Action | ~183 | Xiao Haag flowing narrative with architecture woven in |
| Act 3: Results + Deployment | ~88 | One headline number + Synthea framing + deployment + close |
| **Total** | **~369** | **2:44 at 135 wpm — fits 3:00 with pauses and scroll beats** |
