# MedVision-AI — Dataset Quality Audit

## Dataset
- Root: data/raw/chest_xray
- Classes: NORMAL, PNEUMONIA
- Splits: train, val, test

## Exact duplicate findings within splits
| Split | Class | Duplicate files |
|---|---|---:|
| Train | NORMAL | 1 |
| Train | PNEUMONIA | 25 |
| Validation | NORMAL | 0 |
| Validation | PNEUMONIA | 0 |
| Test | NORMAL | 3 |
| Test | PNEUMONIA | 3 |
| Total | | 32 |

## Exact duplicate findings across splits
| Split comparison | Shared exact image hashes |
|---|---:|
| Train vs Validation | 0 |
| Train vs Test | 0 |
| Validation vs Test | 0 |

## Interpretation
No byte-for-byte identical images were detected across the
training, validation, and test splits. Exact duplicates were
detected within the training and test splits.

This audit does not establish patient-level independence.
Patient-level leakage cannot be ruled out without reliable
patient identifiers or suitable dataset metadata.

## Limitations
- Hash matching detects exact file-content duplicates only.
- Visually similar images with different file contents may not match.
- Filename prefixes alone do not confirm patient identity.
- No files were deleted or modified during this audit.

## Recommendation
Preserve the original dataset and baseline results. If needed,
conduct a separate deduplicated-data experiment and compare
the results with the original baseline.
