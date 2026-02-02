# Common Demographic Errors in Medical Records: Research Report

## Executive Summary

Demographic data errors in medical records are a pervasive problem affecting patient safety, care quality, and healthcare system costs. Patient misidentification errors occur in approximately 1 in 8,000 hospital admissions, contributing to avoidable medical errors and up to $20 billion in annual costs. This report synthesizes current research on the types, frequencies, causes, and impacts of demographic errors to inform realistic error modeling for entity resolution testing.

## 1. Types of Demographic Errors

### 1.1 Name-Related Errors

**Common Error Patterns:**
- Misspelled names (typos, phonetic variations)
- Legal name vs. nicknames
- Middle name vs. middle initial
- Inconsistent handling of:
  - Hyphens in last names
  - Apostrophes in names
  - Suffixes (Jr., Sr., III, etc.)
  - Prefixes (de, von, van, etc.)

**System-Level Variations:**
Some EHR systems include hyphens, apostrophes, and suffixes in last names while others don't, creating matching challenges even with accurate data entry.

### 1.2 Date of Birth Errors

**Common Error Patterns:**
- Transposed digits (e.g., 03/15/1985 vs. 05/13/1985)
- Flipped day/month in international formats
- Incorrect year entry
- Data entry typos at busy registration desks

**Real-World Example:**
Harris County health system (Houston) has 2,488 records with the name "Maria Garcia," of which 231 share the same birth date, illustrating the complexity of matching with common names.

### 1.3 Address-Related Errors

**Common Error Patterns:**
- Outdated addresses (patients move but records aren't updated)
- Misspelled street names
- Inconsistent abbreviations (Street vs. St., Avenue vs. Ave.)
- Missing apartment/unit numbers
- Non-standardized formats vs. USPS standards

**Impact of Standardization:**
Combining addresses in USPS format with last name standardization can improve match rates from 81.3% to 91.6% (a 10 percentage point improvement).

### 1.4 Identifier Errors

**Common Error Patterns:**
- Transposed digits in member IDs or social security numbers
- Wrong policy numbers
- Outdated insurance information
- Incomplete identification numbers

### 1.5 Other Demographic Fields

**Data Quality by Field Type:**
- **High quality:** Age and gender data are usually reliable
- **Moderate to high missing data:** Income, marital status, education, employment status, nationality
- **Often non-standardized:** Race/ethnicity, gender options

## 2. Frequency and Impact

### 2.1 Error Rates

- **Within-organization matching:** 1 in 5 matches can be incorrect (20% error rate)
- **Cross-organization matching:** Match rates can be as low as 50%, even when facilities share the same EHR system
- **Patient harm:** 20% of hospital CIOs reported at least one patient harmed in the past year due to mismatched records

### 2.2 Financial Impact

- **Overall cost:** $20 billion annually in costs from patient misidentification
- **Denied claims:** 35% of denied claims result from inaccurate patient identification
- **Hospital costs:** Average hospital loses $2.5 million annually; US healthcare system loses $6.7 billion
- **Revenue loss from errors:** Preventable denials cost providers 6-8% of revenues

### 2.3 Clinical Impact

- Fragmented patient records
- Misdiagnosis risk
- Incorrect treatment administration
- Potential patient harm
- Treatment delays from missing information
- Adverse reactions from incomplete allergy lists or medication histories

## 3. Root Causes of Demographic Errors

### 3.1 Data Entry Issues

**Primary Contributors (by importance):**
1. **Data entry errors (66%)** - Rated as greatest contributor
2. **Record matching/search terminology (46%)**
3. **Poor system integration (42%)**

**Specific Causes:**
- Manual entry errors at busy registration desks
- Typos and miskeyed fields
- Human error and fatigue
- Lack of standardized data entry protocols
- Varied input methods across systems

### 3.2 System-Level Issues

- Different systems capture demographic information differently
- Data fragmentation across multiple systems
- Multiple systems storing demographics separately without synchronization
- Inconsistent coding standards
- Lack of widespread standardized demographic data elements

### 3.3 Data Maintenance Issues

- Patients move but addresses aren't updated
- Insurance changes not reflected in records
- Name changes (marriage, legal changes) not captured
- Phone numbers and email addresses become outdated
- No routine maintenance of demographic changes

### 3.4 Structural Challenges

- Unstructured documentation
- Missing information during initial data collection
- Patients having multiple "accounts" in systems requiring matching
- Duplicate patient records created from minor variations

## 4. Specific Error Patterns to Model

### 4.1 High-Priority Error Types

Based on the research, the following error types should be prioritized for modeling:

**Tier 1 (Most Common/Impactful):**
- Name misspellings and typos
- Transposed digits in dates of birth
- Transposed digits in member IDs
- Outdated addresses
- Missing middle names or middle initial variations

**Tier 2 (Moderate Frequency):**
- Name formatting variations (hyphens, apostrophes, suffixes)
- Nickname vs. legal name discrepancies
- Address abbreviation inconsistencies
- Phone number format variations

**Tier 3 (Lower Frequency but Important):**
- Missing or incomplete secondary demographic fields
- Gender field variations
- Non-standardized race/ethnicity entries

### 4.2 Realistic Error Distribution

For authentic modeling:
- Apply errors more heavily to address fields (frequent changes)
- Model data entry typos as adjacent key errors or transpositions
- Create systematic variations (e.g., one system always drops hyphens)
- Include temporal aspects (older records have more outdated information)
- Model the 1 in 5 error rate for within-organization matching

### 4.3 Combined Error Scenarios

Real-world scenarios often involve multiple simultaneous errors:
- Name misspelling + outdated address
- Transposed DOB digits + wrong middle initial
- Multiple records with slight variations accumulating over time

## 5. Standards and Best Practices

### 5.1 ONC Patient Demographic Data Quality (PDDQ) Framework

The Office of the National Coordinator for Health Information Technology worked with CMMI Institute to develop the PDDQ Framework specifically to address poor demographic data quality.

### 5.2 Key Standardization Elements

- Standardized telephone number format
- Date of birth format
- USPS address standardization
- Standardized naming conventions
- Gender and race/ethnicity options

### 5.3 AHIMA Recommendations

- Standardized practices in collecting demographic data at registration
- Routine maintenance of demographic changes
- Multiple birth indicators in pediatrics
- Use of additional identifiers or biometrics when possible
- Data quality governance programs

## 6. Implications for Entity Resolution Testing

### 6.1 Recommended Modeling Approach

1. **Base error rates on research findings:**
   - Target 20% error rate for challenging scenarios
   - Include 50% match difficulty for cross-system scenarios

2. **Focus on high-impact fields:**
   - Prioritize name, DOB, and address errors
   - Model realistic typo patterns (adjacent keys, transpositions)

3. **Include system-level variations:**
   - Model how different systems handle special characters
   - Create formatting inconsistencies between data sources

4. **Temporal degradation:**
   - Older records should have higher error rates
   - Model address/phone changes over time

5. **Create duplicate scenarios:**
   - Multiple records for same patient with slight variations
   - Test common name scenarios (e.g., Maria Garcia examples)

### 6.2 Validation Metrics

Your synthetic errors should be evaluated against:
- Do they produce match rates similar to real-world scenarios (50-80%)?
- Do they reflect the reported distribution of error types?
- Can entity resolution algorithms achieve realistic performance?

## 7. References

### Primary Research Sources

1. [Healthcare Data Quality for Patient Safety & Compliance in 2025](https://atlan.com/data-quality-in-healthcare/)
2. [The Importance of Data Quality in Healthcare in 2025](https://kms-healthcare.com/blog/data-quality-in-healthcare/)
3. [Digital Health Data Quality Issues: Systematic Review - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10131725/)
4. [Getting Patient Demographics Accurate - Vozo Blog](https://www.vozohealth.com/blog/getting-patient-demographics-accurate-why-it-matters-for-quality-care-and-reimbursement)
5. [Discovery of data quality issues in electronic health records - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12784561/)

### Patient Matching and EHR Sources

6. [Data-Driven Patient Matching - Vozo Blog](https://www.vozohealth.com/blog/data-driven-patient-matching-overcoming-demographic-data-issues-for-ehr-interoperability)
7. [Electronic Health Records: Patient Matching Challenges - Pew Research](https://www.pew.org/en/research-and-analysis/fact-sheets/2016/11/electronic-health-records-patient-matching-and-data-standardization-remain-top-challenges)
8. [Standardized Demographic Data Improve Patient Matching - Pew Research](https://www.pew.org/en/research-and-analysis/issue-briefs/2019/09/standardized-demographic-data-improve-patient-matching-in-electronic-health-records)
9. [Health Information Technology: Patient Matching - GAO Report](https://www.gao.gov/products/gao-19-197)
10. [Patient Matching in the Era of EHRs - HFMA](https://www.hfma.org/technology/electronic-health-records/62422/)

### Demographic Data Quality Resources

11. [Incomplete or Incorrect Patient Demographics](https://staffingly.com/incomplete-or-incorrect-patient-demographics-preventing-verification-failures/)
12. [Patient Demographic Data Quality Framework - HealthIT.gov](https://www.healthit.gov/playbook/pddq-framework/introduction/)
13. [Best Practices for Data Capture - HealthIT.gov](https://www.healthit.gov/playbook/registrar/chapter-3/)
14. [What's the big deal about patient demographic data? - HealthIT.gov](https://playbook.healthit.gov/playbook/registrar/chapter-1/)

### Standards and Professional Organizations

15. [AHIMA Data Quality and Integrity Policy Statement](https://www.ahima.org/advocacy/policy-statements/data-quality-and-integrity/)
16. [AHIMA Patient Identification Policy](https://www.ahima.org/advocacy/policy-statements/patient-identification/)
17. [AHIMA Public Policy Statement: Data Quality and Integrity (PDF)](https://www.ahima.org/media/3zigqfpy/ahima-data-quality-integrity-public-policy-statement.pdf)
18. [Bipartisan MATCH IT Act of 2025 - AHIMA](https://www.ahima.org/news-publications/press-room-press-releases/2025-press-releases/bipartisan-representatives-reintroduce-match-it-act-of-2025/)

---

**Report prepared:** February 1, 2026
**Research scope:** Common demographic errors in medical records and healthcare data quality
**Intended use:** Informing realistic error modeling for synthetic healthcare data entity resolution testing
