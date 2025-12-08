# Censoring Indicator Conventions in Survival Analysis Adapters

## Overview

This document clarifies the censoring indicator conventions used across different survival analysis packages interfaced in skpro, and how they are translated to and from skpro's internal convention.

## skpro Convention

In skpro, the censoring indicator `C` (passed to `fit()` and other methods) follows this convention:

```
C = 0  => uncensored (event observed)
C = 1  => (right) censored
```

This is documented in:
- `skpro/survival/base.py`
- `skpro/extension_templates/survival.py`
- All survival model docstrings

## External Package Conventions

### lifelines

**Convention**: Uses `event_col` parameter where:
```
event_col = 1  => event observed (uncensored)
event_col = 0  => censored
```

**Source**: 
- lifelines CoxPHFitter.fit() documentation
- lifelines examples and test code

**Transformation in skpro**:
In `skpro/survival/adapters/lifelines.py`, the conversion is:
```python
C_col = 1 - C.copy()  # lifelines uses 1 for uncensored, 0 for censored
```

This is **CORRECT**: 
- skpro C=0 (uncensored) becomes lifelines event_col=1 (uncensored)
- skpro C=1 (censored) becomes lifelines event_col=0 (censored)

### scikit-survival (sksurv)

**Convention**: Uses structured array with `delta` field where:
```
delta = True (1)  => event observed (uncensored)
delta = False (0) => censored
```

**Source**:
- scikit-survival documentation and source code
- Named "delta" indicator following standard survival analysis notation

**Transformation in skpro**:
In `skpro/survival/adapters/sksurv.py`, the conversion is:
```python
C_np_bool = C_np == 0  # sksurv uses "delta" indicator, 0 = censored
# this is the opposite of skpro ("censoring" indicator), where 1 = censored
y_sksurv = list(zip(C_np_bool, y_np))
```

This is **CORRECT**:
- skpro C=0 (uncensored) becomes sksurv delta=True (uncensored)
- skpro C=1 (censored) becomes sksurv delta=False (censored)

### ngboost

**Convention**: Uses `E` (event indicator) where:
```
E = 1  => event observed (uncensored)
E = 0  => censored
```

**Source**:
- ngboost NGBSurvival.fit() signature: `fit(X, T, E)`
- ngboost documentation and examples
- NGBoost evaluation functions and test code

**Transformation in skpro**:
In `skpro/survival/ensemble/_ngboost_surv.py`, the conversion is:
```python
# skpro => 0 = uncensored, 1 = (right) censored
# ngboost => 1 = uncensored, else (right) censored
if C is None:
    C = pd.DataFrame(np.ones(len(y)), index=y.index, columns=y.columns)
else:
    C = 1 - C
```

This is **CORRECT**:
- skpro C=0 (uncensored) becomes ngboost E=1 (uncensored)
- skpro C=1 (censored) becomes ngboost E=0 (censored)

## Summary Table

| Package | Convention | skpro C | Mapped Value | Interpretation |
|---------|-----------|---------|--------------|-----------------|
| **skpro** | C=0 uncensored, C=1 censored | 0 | 0 | Uncensored (event observed) |
| **skpro** | C=0 uncensored, C=1 censored | 1 | 1 | Censored (right-censored) |
| **lifelines** | event_col=1 uncensored, event_col=0 censored | 0 | 1 | Uncensored (event observed) |
| **lifelines** | event_col=1 uncensored, event_col=0 censored | 1 | 0 | Censored (right-censored) |
| **sksurv** | delta=True uncensored, delta=False censored | 0 | True | Uncensored (event observed) |
| **sksurv** | delta=True uncensored, delta=False censored | 1 | False | Censored (right-censored) |
| **ngboost** | E=1 uncensored, E=0 censored | 0 | 1 | Uncensored (event observed) |
| **ngboost** | E=1 uncensored, E=0 censored | 1 | 0 | Censored (right-censored) |

## Conclusion

All three external packages use the **same convention**:
- **1 = event observed/uncensored**
- **0 = censored**

This is the **opposite** of skpro's internal convention:
- **0 = uncensored**
- **1 = censored**

All existing adapters correctly translate between skpro's convention and the external packages' conventions using the transformation `C_external = 1 - C_internal`.

## statsmodels Note

**PR #306** fixed the statsmodels CoxPH adapter to use the correct transformation:
```python
status = 1 - C.to_numpy().flatten() if C is not None else None
```

This follows the same pattern: statsmodels uses `status` where 1 = event observed, 0 = censored.
