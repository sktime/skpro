# Issue #313 Fix Summary

## Issue Description

Issue #313 requested checking and validating that the translations of censoring indicators between interfaced packages (lifelines, scikit-survival, ngboost) and skpro are correct.

The concern was that inconsistency in censoring indicator conventions ("1" can mean either censored or observed, "0" can also mean either) could lead to confusion and bugs.

## Investigation Results

I have thoroughly investigated the censoring indicator conventions across all three packages and verified the correctness of the current implementations.

### Key Finding

All three external packages use the **same convention**:
- **1 = event observed (uncensored)**
- **0 = censored (right-censored)**

This is **opposite** to skpro's internal convention:
- **0 = uncensored**
- **1 = censored**

### Current Implementations - All Correct ✓

#### 1. lifelines Adapter (`skpro/survival/adapters/lifelines.py`)

**Transformation**: `C_lifelines = 1 - C_skpro`

```python
C_col = 1 - C.copy()  # lifelines uses 1 for uncensored, 0 for censored
```

- skpro C=0 (uncensored) → lifelines event_col=1 (uncensored) ✓
- skpro C=1 (censored) → lifelines event_col=0 (censored) ✓

#### 2. scikit-survival Adapter (`skpro/survival/adapters/sksurv.py`)

**Transformation**: `delta = (C_skpro == 0)`

```python
C_np_bool = C_np == 0  # sksurv uses "delta" indicator, 0 = censored
```

- skpro C=0 (uncensored) → sksurv delta=True (uncensored) ✓
- skpro C=1 (censored) → sksurv delta=False (censored) ✓

#### 3. ngboost Adapter (`skpro/survival/ensemble/_ngboost_surv.py`)

**Transformation**: `E_ngboost = 1 - C_skpro`

```python
C = 1 - C  # Convert to ngboost convention where 1 = uncensored
```

- skpro C=0 (uncensored) → ngboost E=1 (uncensored) ✓
- skpro C=1 (censored) → ngboost E=0 (censored) ✓

## Changes Made

### 1. Documentation (`CENSORING_INDICATOR_CONVENTIONS.md`)

Created comprehensive documentation explaining:
- skpro's internal convention
- Each external package's convention
- How transformations map between conventions
- Summary table for quick reference

This document serves as a reference for developers and helps prevent future confusion.

### 2. Code Clarifications

Enhanced comments and docstrings in all three adapters to:
- Explicitly state the censoring convention of each package
- Explain the transformation being applied
- Document the mapping in both directions

This makes the code self-documenting and easier to understand for future maintainers.

**Files modified**:
- `skpro/survival/adapters/lifelines.py`
- `skpro/survival/adapters/sksurv.py`
- `skpro/survival/ensemble/_ngboost_surv.py`

## Verification

- ✓ All files pass Python syntax checks
- ✓ All transformations verified against official package documentation
- ✓ All implementations are mathematically correct
- ✓ No logic changes were needed (all implementations were already correct)

## Branch

All changes were made on branch: `fix/issue-313-censoring-indicator`

## Commits

1. **chore: document censoring indicator conventions for issue #313** - Added comprehensive documentation file
2. **enhancement: clarify censoring indicator conventions in adapter comments** - Enhanced code comments and docstrings

## Conclusion

Issue #313 has been resolved by:

1. **Verifying** that all existing censoring indicator translations are correct
2. **Documenting** the conventions and transformations for future reference
3. **Enhancing** code clarity with explicit comments explaining the conventions

No bugs were found, but the code is now much more transparent about why certain transformations are necessary, which will help prevent future confusion and bugs related to censoring indicator conventions.
