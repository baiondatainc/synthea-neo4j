# US Healthcare Organizations - Sources & Methodology
Generated: 2026-08-13

## Sources (all public domain / free reuse)
- NPPES monthly V2 bulk file - https://download.cms.gov/nppes/NPI_Files.html
- NUCC taxonomy - https://nucc.org
- CMS Hospital Provider Cost Report (HCRIS) - https://data.cms.gov
- CMS Provider of Services (POS, QIES) - https://data.cms.gov
- CMS Hospital Enrollments (NPI-CCN crosswalk) - https://data.cms.gov
- AHRQ Compendium (optional) - https://www.ahrq.gov/chsp

## Method
NPPES filtered to Entity Type 2 (organizations), active NPIs only;
taxonomy mapped to subtype via NUCC; deduplicated on normalized
legal name + state + ZIP; hospital beds joined via CCN crosswalk
to HCRIS cost reports; SNF/other facility beds from POS QIES;
size: Large >= 300 beds, Medium >= 100, else Small;
no bed data -> Unclassified.

## Known limitations
- NPPES is self-reported; some records are stale.
- Non-Medicare providers lack bed/size data (Unclassified).
- Size for non-hospital facilities requires the FFS enrollment
  crosswalk (ffs_enrollment.csv); without it only hospitals size.
- TPAs / self-insured plans not covered by CMS payer files.

## Sanity checks (expected: ~6-15k hospitals depending on
## subtype breadth; ~4-6k hospitals with beds)
- Total organizations: 1,735,190
- Hospitals: 14,962
- Payers: 10,674
- With bed data: 5,665
- Large: 742
- Medium: 1,568
- Small: 3,355
- Hospitals with beds: 2,671