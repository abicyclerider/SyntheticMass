# Future Enhancements Roadmap

## Current Status

The Entity Resolution Data Augmentation System is **production-ready** with baseline functionality:
- ✅ Multi-facility distribution (1-5+ facilities per patient)
- ✅ Chronological encounter splitting
- ✅ Baseline demographic error types
- ✅ Ground truth tracking
- ✅ Full test coverage
- ✅ Validated on 571 patients, 45,451 encounters

## Planned Enhancements

### 1. Research-Informed Error Types (High Priority)

**Current State**: Error types are baseline implementations of common data quality issues (name variations, address errors, typos, etc.)

**Enhancement**: Refine error types based on actual research findings in `config/research/error_patterns.md`

**Steps**:
1. Review `config/research/error_patterns.md` for identified patterns
2. Compare against current error implementations
3. Update error type weights in `config/default_config.yaml`
4. Implement new error classes for any missing patterns
5. Re-validate with updated error distribution

**Impact**: More realistic error patterns reflecting actual healthcare data quality issues

---

### 2. Temporal Encounter Clustering (Medium Priority)

**Enhancement**: Group related encounters by medical condition at same facility

**Example**: All encounters for diabetes treatment occur at the same endocrinology clinic, simulating patient care continuity

**Implementation**:
- Analyze encounter diagnosis codes
- Cluster encounters by condition type
- Ensure clustered encounters stay at same facility

**Impact**: More realistic patient-facility relationships

---

### 3. Facility Specialization (Medium Priority)

**Enhancement**: Model facility types (general hospital, specialty clinic, urgent care)

**Example**: Cardiology procedures more likely at cardiac centers

**Implementation**:
- Assign specialty types to facilities
- Weight encounter distribution by procedure/condition type
- Update FacilityGenerator to create specialized facilities

**Impact**: More realistic healthcare system modeling

---

### 4. Longitudinal Error Evolution (Low Priority)

**Enhancement**: Errors change over time (address changes are permanent, typos are random)

**Example**:
- Patient moves → all future encounters have new address
- Typo → random per facility

**Implementation**:
- Add temporal dimension to error tracking
- Persistent vs transient error types
- Update ErrorInjector to consider encounter dates

**Impact**: More nuanced error patterns

---

### 5. Performance Optimization (Low Priority)

**Current Performance**: ~0.08 seconds per patient (4 seconds for 571 patients)

**Potential Optimizations**:
- Parallel processing for independent facilities
- Streaming CSV processing for very large datasets
- Caching for reference table lookups

**Trigger**: Only needed if processing >10,000 patients

---

## Non-Goals

- ❌ Real-time data generation (batch processing is sufficient)
- ❌ Database integration (CSV-based is fine for intended use case)
- ❌ GUI interface (CLI is sufficient for research/competition use)
- ❌ Distributed computing (single-machine processing is adequate)

## Contributing

To implement an enhancement:
1. Create feature branch from `main`
2. Update relevant modules (see Architecture in README.md)
3. Add unit tests following Sandi Metz principles
4. Update documentation
5. Submit PR with test results

## Questions?

See `IMPLEMENTATION_SUMMARY.md` for technical details or `QUICK_START.md` for usage examples.
