# Entity Resolution Blocking Strategy Research Context

## Problem Statement

I'm implementing entity resolution for synthetic medical records to match patient records across multiple healthcare facilities. The blocking strategies tested so far are **missing too many true matches**, resulting in unacceptable recall rates.

**Goal:** Find blocking strategies that achieve **≥95% recall** while maintaining reasonable computational efficiency (candidate pair reduction).

---

## Dataset Characteristics

**Source:** SyntheticMass augmented patient data
- **Run ID:** `run_20260203_071928`
- **Total records:** 1,228 patient records across 5 facilities
- **True unique patients:** 571
- **True matching pairs:** 1,121 (pairs of records that are the same patient)
- **Average records per patient:** 2.15 facilities per patient
- **Error rate:** 42.6% of patients have demographic errors

### Error Profile (Intentionally Injected)

| Field | Error Count | Error Types |
|-------|-------------|-------------|
| BIRTHDATE | 75 | ±1 day off-by-one |
| ADDRESS | 86 | Abbreviations (St./Street), format variations, apartment formatting |
| SSN | 53 | Transpositions, digit errors, format variations |
| NAMES | 64 | Typos, capitalization (SMITH vs Smith), maiden name usage |

### Key Fields Available for Blocking

- **FIRST** (first name) - subject to typos, capitalization errors
- **LAST** (last name) - subject to typos, capitalization errors, maiden name switches
- **MAIDEN** (maiden name) - often null
- **SSN** - 9-digit, subject to transpositions/errors
- **BIRTHDATE** - date field, subject to ±1 day errors
- **ADDRESS** - full street address, highly variable formatting
- **CITY** - city name
- **STATE** - 2-letter state code (e.g., MA, NY)
- **ZIP** - 5-digit ZIP code
- **GENDER** - M/F

---

## Current Blocking Results

### Simple Blocking Strategies (TESTED)

| Strategy | Candidate Pairs | True Pairs Found | Recall | Reduction Rate | Issue |
|----------|----------------|------------------|---------|----------------|-------|
| **lastname_state** | 2,059 | 857 / 1,121 | **76.4%** | 99.7% | ❌ Missing 24% of matches |
| **lastname_only** | 2,059 | 857 / 1,121 | **76.4%** | 99.7% | ❌ Same as above |
| **state_only** | 753,378 | 1,121 / 1,121 | 100% | 0% | ❌ No reduction (useless) |
| **zip_state** | 59,825 | 1,121 / 1,121 | 100% | 92.1% | ⚠️ Too many pairs |

### Advanced Strategies (TESTED)

| Strategy | Window/Config | Recall | Notes |
|----------|---------------|---------|-------|
| **sorted_neighborhood** | window=3,5,7 on LAST | ~80-85% | Still missing matches |
| **multipass (lastname+sorted)** | combo | ~85-90% | Better but not enough |

**Problem:** Even sorted neighborhood and multipass strategies are missing 10-15% of true matches due to high error rates in blocking keys.

---

## Root Cause Analysis

**Why are we missing matches?**

1. **Name errors prevent exact/phonetic blocking:**
   - "Smith" vs "Smyth" (typo)
   - "JOHNSON" vs "Johnson" (case mismatch)
   - "Brown" vs "Smith" (maiden name change)

2. **High error rate (42.6%) means blocking keys are unreliable:**
   - Can't trust exact matching on any single field
   - Multiple fields may have errors simultaneously

3. **Sorted neighborhood helps but has limitations:**
   - Window size trade-off: small window misses matches, large window = too many pairs
   - Only works on one field at a time
   - Doesn't handle maiden name switches

---

## Constraints & Requirements

### Hard Requirements
- **Recall ≥ 95%:** Must find at least 95% of true matches (1,065+ out of 1,121)
- **Candidate pairs < 100,000:** Computational feasibility for comparison step
- **No use of UUIDs:** Must rely only on demographic fields (realistic scenario)

### Nice to Have
- Reduction rate > 85% (fewer pairs = faster comparison)
- Interpretable/explainable strategy
- Generalizable to real-world data (not overfitted to synthetic errors)

---

## Technologies Available

- **Python 3.12**
- **recordlinkage 0.16** (Python Record Linkage Toolkit)
- **pandas 2.3.3**
- **jellyfish 1.2.1** (string similarity: Jaro-Winkler, Levenshtein, soundex, metaphone)
- **scikit-learn 1.8.0** (for ML-based blocking if needed)

### recordlinkage Blocking Methods Available
```python
import recordlinkage as rl
indexer = rl.Index()

# Available methods:
indexer.block(left_on='field')                    # Exact match blocking
indexer.sortedneighbourhood(left_on='field', window=N)  # Sorted neighborhood
indexer.full()                                    # All pairs (no blocking)
```

---

## What I Need Help With

**Primary Question:**
> What blocking strategies can achieve ≥95% recall on entity resolution with 42.6% error rate in blocking keys?

**Specific areas to explore:**

1. **Multi-field blocking strategies:**
   - How to combine multiple blocking keys effectively?
   - Union vs intersection of blocking results?

2. **Fuzzy/approximate blocking:**
   - Phonetic encoding (soundex, metaphone, NYSIIS) for names?
   - Blocking on string similarity bins rather than exact matches?
   - Locality-sensitive hashing (LSH) for approximate matching?

3. **Adaptive/learned blocking:**
   - ML-based blocking (train classifier to predict candidate pairs)?
   - Canopy clustering for blocking?

4. **Multi-pass strategies:**
   - Optimal combination of passes to maximize recall?
   - How to balance number of passes vs pair explosion?

5. **Alternative approaches:**
   - Should I skip traditional blocking and use different paradigm?
   - Hierarchical/iterative matching?
   - Graph-based approaches?

---

## Code Example (How Blocking is Used)

```python
import pandas as pd
import recordlinkage as rl

# Load patient data (1,228 records)
patients = pd.read_csv('patients.csv')
patients_indexed = patients.set_index('record_id')

# Current approach (lastname_state blocking)
indexer = rl.Index()
indexer.block(left_on=['LAST', 'STATE'])
candidate_pairs = indexer.index(patients_indexed)
# Result: 2,059 pairs, but only finds 857/1,121 true matches (76.4% recall)

# Next step: compare these pairs using string similarity
compare = rl.Compare()
compare.string('FIRST', 'FIRST', method='jarowinkler')
compare.string('LAST', 'LAST', method='jarowinkler')
# ... more comparisons
features = compare.compute(candidate_pairs, patients_indexed)

# Then classify as match/non-match
# But if blocking missed 264 true pairs, they'll never be recovered!
```

---

## Example True Match That's Being Missed

**Record 1 (facility_001):**
```
FIRST: Nathan
LAST: Krajcik
SSN: 999-20-7076
BIRTHDATE: 1944-02-06
ADDRESS: 241 THOMPSON LN.
STATE: MA
```

**Record 2 (facility_002):**
```
FIRST: Nathan
LAST: Krajcik
SSN: 999-20-7706
BIRTHDATE: 1943-02-06
ADDRESS: 241 Thompson Lane
STATE: MA
```

**Why missed by lastname_state?**
- Both have LAST="Krajcik" and STATE="MA", so they SHOULD be in same block
- But if there's a typo like "Krajcik" vs "Krajick", exact blocking misses them

**Why missed by sorted_neighborhood?**
- If window is too small and records are far apart alphabetically
- Or if maiden name was used at one facility

---

## Questions for Research Thread

1. **Literature:** What do research papers recommend for blocking with high error rates?
2. **Best practices:** What blocking strategies are used in real-world entity resolution systems (hospitals, credit bureaus, etc.)?
3. **recordlinkage limitations:** Are we hitting fundamental limits of the library? Should we use a different tool?
4. **Trade-off validation:** Is ≥95% recall with <100k pairs even feasible with 42.6% error rate?
5. **Hybrid approaches:** Should we combine traditional blocking with ML-based candidate generation?

---

## References

- Project: SyntheticMass entity resolution system
- Location: `/Users/alex/repos/Kaggle/SyntheticMass/`
- Notebook: `analysis/notebooks/entity_resolution_exploration.ipynb`
- Ground truth: `output/augmented/run_20260203_071928/metadata/ground_truth.csv`
- Python Record Linkage Toolkit docs: https://recordlinkage.readthedocs.io/

---

## Success Criteria for New Approach

A blocking strategy that achieves:
- ✅ **Recall ≥ 95%** (find 1,065+ of 1,121 true pairs)
- ✅ **Candidate pairs < 100,000** (computationally feasible)
- ✅ **Reduction rate > 85%** (efficiency)
- ✅ **Implementable with recordlinkage or standard Python libraries**
- ✅ **Generalizable** (not overfitted to synthetic data patterns)

**Current best:** zip_state with 100% recall but 59,825 pairs (acceptable but could be better)

**Goal:** Find a strategy that maintains high recall while reducing pairs further, or validate that zip_state is optimal given the constraints.
