# Health Data Analyst Portfolio — UHIN Application

Oct 25, 2026 

## Portfolio overview

Seven end-to-end projects that prove every core skill in the UHIN Health Data Analyst posting, then go one level deeper with causal inference, explainable ML, clinical NLP, and automated data-quality monitoring. Each project uses public or synthetic data only, so no real PHI ever touches the portfolio.

**Positioning statement (for the top of your GitHub README and resume):** Health data analyst who turns messy claims and clinical feeds into trusted, decision-ready measures. I build the SQL, the quality checks, the risk and quality models, and the dashboards, then explain the result to clinicians and executives in plain language.

### How each project maps to the job description

| # | Project | JD requirement it proves | Advanced technique added |
| --- | --- | --- | --- |
| 1 | Multi-source data quality framework | Data integrity & QA; completeness, accuracy, timeliness; coordinating with data contributors | Automated expectation suites, statistical feed-anomaly detection, master patient index matching |
| 2 | Claims utilization & cost analytics in SQL | Advanced, optimized SQL; readmissions, LOS, total cost of care | Window functions, episode grouping, gaps-and-islands, query plans and indexing |
| 3 | Risk adjustment + readmission prediction | CMS-HCC risk stratification; predictive risk scoring with the CDO | Gradient boosting, SHAP explanations, calibration, fairness audit |
| 4 | Quality measures engine | HEDIS, MIPS, CMS Star Ratings; quality-of-care tracking | Measure-as-code, care-gap lists, statistical process control charts |
| 5 | Value-based care program evaluation | Evaluating VBC initiatives; statistical and research methods | Propensity score matching, difference-in-differences, interrupted time series |
| 6 | Clinical NLP on unstructured notes | Unstructured data; NLP (preferred); healthcare data standards | Negation-aware entity extraction, SDOH detection, FHIR mapping |
| 7 | Executive & clinical dashboards | Tableau / Power BI; KPIs for readmission, LOS, cost | Drill-through design, DAX time intelligence, small-cell suppression |
| — | Governance & documentation pack | HIPAA, CMS, data governance; SOPs, report specs, training | Safe Harbor de-identification code, data dictionary, report spec template |

### Data sources (all free, no PHI)

- **Synthea** synthetic patients: full clinical records (encounters, conditions, labs, meds) in CSV and FHIR. Use the Utah demographic module to make it locally relevant.
- **CMS DE-SynPUF**: synthetic Medicare claims (inpatient, outpatient, carrier, Part D, beneficiary) — the classic claims practice set.
- **CMS public files**: HCC model software and ICD-10 crosswalks, Hospital Readmissions Reduction Program data, Care Compare / Star Ratings files.
- **MIMIC-IV demo** or **MTSamples**: de-identified clinical notes for NLP (follow each dataset’s license terms).

### Tech stack

SQL (PostgreSQL locally, Google BigQuery for scale) · Python (pandas, polars, statsmodels, scikit-learn, XGBoost, SHAP, spaCy/medspaCy, matplotlib, plotly) · R (MatchIt, did) for one causal analysis · Tableau Public and Power BI · dbt for transformation and tests · Git/GitHub.

> Results in this portfolio are written as fill-in templates like \[X%\]. Run the code on the data, then replace each bracket with your real number. Never publish a number you did not produce.

## Project 1 — Multi-source health data quality framework

**Goal:** Score every incoming clinical and claims feed on the four quality dimensions UHIN names (accuracy, completeness, timeliness, consistency), flag anomalies automatically, and produce a contributor-facing scorecard. This mirrors QA on an HIE such as the CHIE, where dozens of hospitals and payers submit data of uneven quality.

**Business question:** Which data contributors are sending incomplete, late, or invalid data, and is any feed suddenly breaking?

### Quality dimensions and checks

| Dimension | Example check | Threshold |
| --- | --- | --- |
| Completeness | % of encounters with a valid principal diagnosis; % of patients with DOB, sex, ZIP | ≥ 98% |
| Conformance / validity | ICD-10 codes exist in the code set for the service date; NPI passes Luhn check; dates parse | 100% |
| Plausibility / accuracy | Discharge ≥ admit date; age 0–115; pregnancy codes only on plausible sex/age; lab values in physiologic range | ≥ 99.5% |
| Timeliness | Median lag from service date to receipt; % of ADT messages within 24 hours | Median ≤ 2 days |
| Consistency | Same patient’s DOB and sex match across sources; claim totals equal sum of lines | ≥ 99% |
| Uniqueness | Duplicate claims (same member, provider, date, code, amount) | ≤ 0.1% |

### SQL: completeness and validity scorecard by contributor

```sql
WITH enc AS (
  SELECT
    e.source_org_id,
    e.encounter_id,
    (e.principal_dx IS NOT NULL)                         AS has_dx,
    (icd.code IS NOT NULL)                               AS dx_valid,
    (e.discharge_dt >= e.admit_dt)                       AS dates_plausible,
    (e.received_ts::date - e.discharge_dt)               AS lag_days
  FROM stg.encounters e
  LEFT JOIN ref.icd10cm icd
         ON icd.code = e.principal_dx
        AND e.discharge_dt BETWEEN icd.effective_start AND icd.effective_end
  WHERE e.received_ts >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT
  source_org_id,
  COUNT(*)                                               AS encounters,
  ROUND(100.0 * AVG(has_dx::int), 2)                     AS pct_dx_present,
  ROUND(100.0 * AVG(dx_valid::int), 2)                   AS pct_dx_valid,
  ROUND(100.0 * AVG(dates_plausible::int), 2)            AS pct_dates_plausible,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lag_days)  AS median_lag_days
FROM enc
GROUP BY source_org_id
ORDER BY pct_dx_valid ASC;
```

### Python: automated feed-anomaly detection (advanced)

Daily record counts per contributor are checked against their own history with a robust z-score (median and MAD, so one past outage does not distort the baseline) and a day-of-week adjustment.

```python
import pandas as pd
import numpy as np

def flag_volume_anomalies(daily: pd.DataFrame, window: int = 56, z_cut: float = 3.5) -> pd.DataFrame:
    """daily: columns source_org_id, date, n_records."""
    daily = daily.sort_values(["source_org_id", "date"]).copy()
    daily["dow"] = pd.to_datetime(daily["date"]).dt.dayofweek

    def score(g):
        # compare each day only with the same weekday in the trailing window
        med = g.groupby("dow")["n_records"].transform(
            lambda s: s.shift(1).rolling(window // 7, min_periods=4).median())
        mad = g.groupby("dow")["n_records"].transform(
            lambda s: (s.shift(1) - s.shift(1).rolling(window // 7, min_periods=4).median())
                      .abs().rolling(window // 7, min_periods=4).median())
        g["robust_z"] = 0.6745 * (g["n_records"] - med) / mad.replace(0, np.nan)
        return g

    out = daily.groupby("source_org_id", group_keys=False).apply(score)
    out["anomaly"] = out["robust_z"].abs() > z_cut
    return out[out["anomaly"]]
```

### Advanced add-ons

- **Expectation suites as code:** encode the checks above as dbt tests or Great Expectations suites so they run on every load and fail loudly instead of being run by hand.
- **Patient matching:** a probabilistic record-linkage step (Fellegi-Sunter weights via the `splink` library) on name, DOB, sex and ZIP, reporting the estimated duplicate-patient rate per contributor.
- **Code-set drift:** a chi-square test comparing this month’s diagnosis-code distribution with the trailing 6 months, to catch a contributor that silently changed its coding or mapping.

### Deliverables

- Contributor scorecard (dashboard tab in Project 7) with red/amber/green status per dimension.
- An anomaly email template addressed to a data contributor: what broke, since when, example record IDs (no PHI in the email), and the fix requested.
- **Result template:** Checks found \[N\] contributors below the 98% completeness threshold; the anomaly detector caught a simulated \[X%\] volume drop on day \[N\] of the outage.

## Project 2 — Claims utilization and cost analytics in SQL

**Goal:** Build the core utilization and cost measures named in the posting — 30-day readmission rate, length of stay, and total cost of care — directly from raw claims, with production-quality SQL.

**Business question:** Which conditions, facilities and member segments drive avoidable readmissions and cost per member per month (PMPM)?

### Step 1 — Continuous enrollment with gaps-and-islands

Most measures require continuous enrollment. This collapses monthly eligibility rows into continuous spans; a second pass then applies each measure’s allowable gap (HEDIS commonly allows one gap of up to 45 days).

```sql
WITH m AS (
  SELECT member_id, elig_month,
         elig_month - (ROW_NUMBER() OVER (PARTITION BY member_id ORDER BY elig_month)
                       * INTERVAL '1 month') AS island_key
  FROM stg.eligibility_monthly
),
spans AS (
  SELECT member_id,
         MIN(elig_month)                                   AS span_start,
         (MAX(elig_month) + INTERVAL '1 month - 1 day')::date AS span_end
  FROM m
  GROUP BY member_id, island_key
)
SELECT * FROM spans;
```

### Step 2 — Index admissions and 30-day all-cause readmissions

Transfers are merged into one stay first (discharge and next admit within 1 day), otherwise a hospital transfer is wrongly counted as a readmission.

```sql
WITH ip AS (   -- one row per inpatient claim header
  SELECT member_id, claim_id, admit_dt, discharge_dt, discharge_status, drg, paid_amt
  FROM stg.claims_inpatient
),
flagged AS (   -- start a new stay unless this admit is a transfer from the prior discharge
  SELECT *,
         CASE WHEN admit_dt <= LAG(discharge_dt) OVER w + 1 THEN 0 ELSE 1 END AS new_stay
  FROM ip
  WINDOW w AS (PARTITION BY member_id ORDER BY admit_dt, discharge_dt)
),
stays AS (
  SELECT member_id,
         SUM(new_stay) OVER (PARTITION BY member_id ORDER BY admit_dt, discharge_dt) AS stay_no,
         admit_dt, discharge_dt, discharge_status, drg, paid_amt
  FROM flagged
),
stay_level AS (
  SELECT member_id, stay_no,
         MIN(admit_dt) AS admit_dt, MAX(discharge_dt) AS discharge_dt,
         (ARRAY_AGG(discharge_status ORDER BY discharge_dt DESC))[1] AS final_status,
         (ARRAY_AGG(drg ORDER BY admit_dt))[1] AS drg,
         SUM(paid_amt) AS stay_paid,
         MAX(discharge_dt) - MIN(admit_dt) AS los_days
  FROM stays
  GROUP BY member_id, stay_no
),
with_next AS (
  SELECT *,
         LEAD(admit_dt) OVER (PARTITION BY member_id ORDER BY admit_dt) AS next_admit_dt
  FROM stay_level
)
SELECT
  drg,
  COUNT(*)                                                         AS index_stays,
  ROUND(AVG(los_days), 2)                                          AS avg_los,
  ROUND(100.0 * AVG((next_admit_dt - discharge_dt <= 30)::int), 2) AS readmit_30d_pct,
  ROUND(AVG(stay_paid), 0)                                         AS avg_paid
FROM with_next
WHERE final_status NOT IN ('20', '07')   -- exclude deaths and left against medical advice
GROUP BY drg
HAVING COUNT(*) >= 30                    -- suppress unstable small cells
ORDER BY readmit_30d_pct DESC;
```

### Step 3 — Total cost of care PMPM

```sql
SELECT
  DATE_TRUNC('month', c.service_dt) AS month,
  c.setting,                              -- IP, OP, professional, Rx, ED
  SUM(c.paid_amt) / NULLIF(mm.member_months, 0) AS pmpm
FROM mart.claims_all c
JOIN mart.member_months mm
  ON mm.month = DATE_TRUNC('month', c.service_dt)
GROUP BY 1, 2, mm.member_months
ORDER BY 1, 2;
```

### Advanced techniques to show

- **Expected vs observed:** join each stay to its risk score from Project 3 and report the observed-to-expected (O/E) readmission ratio by facility, so hospitals with sicker patients are not unfairly penalized.
- **Episode grouping:** bundle all claims from 3 days before admission to 90 days after discharge to measure episode cost for joint replacement or heart failure, the logic behind bundled-payment programs.
- **Performance tuning:** include the `EXPLAIN ANALYZE` output before and after adding a composite index on `(member_id, admit_dt)` and partitioning the claims table by service month. In BigQuery, show partitioning plus clustering on `member_id` and the bytes-scanned reduction.
- **Testing:** dbt tests that assert `discharge_dt >= admit_dt`, unique `claim_id`, and readmission rates within a plausible 5–25% band.

### Result template

Across \[N\] index stays, the 30-day all-cause readmission rate was \[X%\]; heart failure and COPD DRGs ran \[X\] points above average and drove \[X%\] of readmission spend. Adding the composite index cut query runtime from \[X\] s to \[X\] s.

## Project 3 — Risk adjustment and predictive readmission model

**Goal:** Reproduce CMS-HCC risk scores from claims diagnoses, stratify the population into care-management tiers, then build an explainable machine-learning model that predicts 30-day readmission — the “clean datasets for predictive risk-scoring” work the posting describes with the CDO.

**Business question:** Which members should care managers call this week, and why?

### Part A — CMS-HCC risk scoring

1. Map each member’s ICD-10 diagnoses from eligible claim types (inpatient, outpatient, professional face-to-face) to HCCs using the CMS crosswalk for the payment year (V28 for current Medicare Advantage).
2. Apply hierarchies: within a disease family keep only the most severe HCC (for example, diabetes with complications drops diabetes without complications).
3. Sum the demographic factor (age/sex, Medicaid, originally-disabled), the HCC coefficients, and interaction terms (for example, diabetes × heart failure).
4. Normalize, and validate against the CMS reference software on a sample of members — matching to two decimals is the acceptance test.

```sql
-- hierarchy step: drop an HCC when a more severe HCC in the same family is present
SELECT h.member_id, h.hcc
FROM member_hccs h
WHERE NOT EXISTS (
  SELECT 1
  FROM member_hccs h2
  JOIN ref.hcc_hierarchy r
    ON r.parent_hcc = h2.hcc AND r.child_hcc = h.hcc
  WHERE h2.member_id = h.member_id
);
```

**Stratification tiers:** top 1% (complex case management), next 4% (high risk), next 15% (rising risk), remaining 80% (prevention and wellness). Report each tier’s share of members versus share of total cost — typically the top 5% of members account for roughly half of spending.

**Advanced add-on — coding-gap detection:** flag members with evidence of a chronic condition (insulin fill, dialysis procedure, prior-year HCC) but no qualifying diagnosis this year. These “suspected HCC” lists support accurate documentation, not upcoding, and that distinction should be stated plainly.

### Part B — Machine-learning readmission model

Features: HCC risk score and flags, prior 12-month admissions and ED visits, LOS of the index stay, discharge disposition, polypharmacy count, recent lab abnormalities (from clinical data), and an area deprivation index from ZIP.

```python
import xgboost as xgb, shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

# temporal split: train on 2022-2023 discharges, test on 2024 (no future leakage)
train, test = df[df.discharge_year < 2024], df[df.discharge_year == 2024]
X_tr, y_tr = train[FEATURES], train["readmit_30d"]
X_te, y_te = test[FEATURES], test["readmit_30d"]

base = xgb.XGBClassifier(
    n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
    eval_metric="aucpr")
# GroupKFold keeps all of one member's stays in the same fold
model = CalibratedClassifierCV(base, method="isotonic",
                               cv=GroupKFold(5).split(X_tr, y_tr, train["member_id"]))
model.fit(X_tr, y_tr)

p = model.predict_proba(X_te)[:, 1]
print("AUROC", roc_auc_score(y_te, p), "AUPRC", average_precision_score(y_te, p),
      "Brier", brier_score_loss(y_te, p))

# explanations from the uncalibrated booster refit on all training data
booster = base.fit(X_tr, y_tr)
explainer = shap.TreeExplainer(booster)
shap.summary_plot(explainer.shap_values(X_te), X_te)
```

### What makes this advanced

- **Benchmark first:** compare against the LACE index and a logistic regression. A complex model must beat both to justify itself.
- **Calibration:** show a reliability curve. Care managers act on “this member has a 32% risk”, so the probability must be honest, not just well-ranked.
- **Operational metric:** precision among the top 5% of scores (the people care managers can actually call), not only AUROC.
- **Explainability:** per-member SHAP “top 3 reasons” shown beside each name on the outreach list.
- **Fairness audit:** compare calibration and false-negative rates by sex, age band, rural/urban and payer type; report any gap and the mitigation.
- **Model card:** intended use, training data, performance, limits, monitoring plan (monthly drift check on score distribution and outcome rate).

### Result template

The XGBoost model reached AUROC \[X\] versus \[X\] for LACE; among the top 5% of scores, \[X%\] were readmitted, \[X\]× the base rate. Calibration error stayed below \[X\] points in every subgroup except \[group\], which is flagged for review.

## Project 4 — Quality measures engine (HEDIS, MIPS, Star Ratings)

**Goal:** Compute a set of quality measures from combined claims and clinical data, produce provider-level care-gap lists, and monitor performance over time with statistical process control.

**Business question:** Where are we losing quality points, which patients close the gap fastest, and is a change in rate real or just noise?

### Measures implemented

| Measure | Program | Denominator (simplified) | Numerator (simplified) |
| --- | --- | --- | --- |
| Glycemic Status Assessment, HbA1c > 9% (GSD) | HEDIS / Stars | Members 18–75 with diabetes | Most recent HbA1c in the year > 9% or missing (inverse measure) |
| Controlling High Blood Pressure (CBP) | HEDIS / Stars / MIPS | Members 18–85 with hypertension | Most recent BP < 140/90 |
| Breast Cancer Screening (BCS-E) | HEDIS / Stars | Women 50–74 | Mammogram in the prior 27 months |
| Plan All-Cause Readmissions (PCR) | HEDIS / Stars | Index stays 18+ | Readmission within 30 days, reported as observed/expected |
| Statin Use in Persons with Diabetes | Stars (Part D) | Members 40–75 with diabetes fills | ≥ 1 statin fill |

I will always cite the measurement-year specification I followed; real measure logic includes exclusions (hospice, palliative care, frailty plus advanced illness) that must be built, not skipped.

### SQL: measure-as-code pattern

Every measure produces one row per member with denominator, exclusion and numerator flags, so rates, care gaps and audits come from the same table.

```sql
WITH denom AS (
  SELECT p.member_id
  FROM mart.members p
  JOIN mart.continuous_enrollment ce USING (member_id)
  WHERE ce.measurement_year = 2025
    AND DATE_PART('year', AGE(DATE '2025-12-31', p.dob)) BETWEEN 18 AND 75
    AND EXISTS (SELECT 1 FROM mart.dx d
                JOIN ref.value_sets v ON v.code = d.code AND v.value_set = 'Diabetes'
                WHERE d.member_id = p.member_id AND d.service_dt BETWEEN '2024-01-01' AND '2025-12-31')
),
excl AS (
  SELECT DISTINCT member_id FROM mart.dx d
  JOIN ref.value_sets v ON v.code = d.code AND v.value_set IN ('Hospice', 'Palliative Care')
  WHERE d.service_dt BETWEEN '2025-01-01' AND '2025-12-31'
),
last_a1c AS (
  SELECT DISTINCT ON (member_id) member_id, result_value, result_dt
  FROM mart.labs
  WHERE loinc IN ('4548-4', '17856-6') AND result_dt BETWEEN '2025-01-01' AND '2025-12-31'
  ORDER BY member_id, result_dt DESC
)
SELECT d.member_id,
       1                                   AS in_denominator,
       (e.member_id IS NOT NULL)::int      AS excluded,
       (a.result_value IS NULL OR a.result_value > 9)::int AS poor_control   -- inverse: lower is better
FROM denom d
LEFT JOIN excl e USING (member_id)
LEFT JOIN last_a1c a USING (member_id);
```

### Advanced techniques

- **Care-gap prioritization:** rank open gaps by (Star weight of the measure) × (probability the gap closes, from a simple model on past outreach) so outreach staff work the most valuable list first.
- **Statistical process control:** a monthly p-chart per measure with 3-sigma limits; only points outside the limits or runs of 8 on one side are called real changes. This stops leadership from reacting to noise.
- **Confidence intervals for small practices:** Wilson intervals on each clinic’s rate, plus empirical Bayes shrinkage toward the network mean, so a 4-patient clinic is not ranked “best” or “worst” by chance.
- **Star cut-point simulation:** estimate how many additional compliant members are needed to move a measure up one Star level.
- **Hybrid data value:** show how much each rate rises when clinical lab and vitals data are added to claims-only data — the core value of an HIE like UHIN.

### Result template

Adding clinical data from the HIE raised the CBP rate from \[X%\] (claims only) to \[X%\]. Closing \[N\] prioritized gaps would lift BCS from \[X\] to \[X\] Stars.

## Project 5 — Value-based care program evaluation (causal inference)

**Goal:** Answer “did the care-management program actually work?” with research-grade methods instead of a naive before-and-after comparison. This is the strongest differentiator in the portfolio: most candidates report correlations; this project estimates effects.

**Scenario:** A transitional-care program enrolled high-risk heart failure and COPD patients after discharge at some clinics starting in a given quarter. Outcomes: 30-day readmissions, ED visits, and total cost of care PMPM over 6 months.

**Why the naive answer is wrong:** enrolled patients were selected because they were high-risk, and high-cost patients naturally drift back toward average (regression to the mean). A raw before/after drop overstates savings.

### Method 1 — Propensity score matching (R)

```r
library(MatchIt); library(cobalt); library(marginaleffects)

m <- matchit(enrolled ~ age + sex + hcc_score + prior_ip_12m + prior_ed_12m +
               baseline_pmpm + adi_decile + rural,
             data = cohort, method = "nearest", distance = "glm",
             caliper = 0.2, ratio = 2)
love.plot(m, thresholds = c(m = 0.1))    # every standardized mean difference < 0.1

md  <- match.data(m)
fit <- glm(readmit_30d ~ enrolled + hcc_score + prior_ip_12m,
           data = md, weights = weights, family = binomial)
avg_comparisons(fit, variables = "enrolled", vcov = ~subclass, wts = "weights")
```

### Method 2 — Difference-in-differences (Python)

Compare the change in PMPM at program clinics with the change at comparison clinics over the same period, removing shared trends.

```python
import statsmodels.formula.api as smf

# panel: one row per member-month; post = after launch; treated = program clinic
mod = smf.ols("pmpm ~ treated * post + C(month) + hcc_score", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["clinic_id"]})
print(mod.summary().tables[1])          # treated:post = DiD effect in $ PMPM
```

Show the event-study plot (effect by month relative to launch) to check parallel pre-trends. If clinics launched at different times, use the Callaway–Sant’Anna estimator (R `did` package) instead of plain two-way fixed effects.

### Method 3 — Interrupted time series

For a network-wide change with no control group (for example, a new discharge-follow-up policy), fit segmented regression on the monthly readmission rate with level and slope change terms, Newey-West errors for autocorrelation, and a seasonality term.

### Robustness checks to report

- E-value for unmeasured confounding (how strong a hidden confounder would need to be to erase the effect).
- Placebo test: apply the “launch” one year earlier; the estimated effect should be near zero.
- Bootstrap 95% confidence intervals on the dollar savings.
- Return on investment: program cost per enrollee versus estimated avoided cost, with the uncertainty range, not a single point.

### Result template

After matching (\[N\] enrolled, \[N\] controls, all SMDs < 0.1), the program reduced 30-day readmissions by \[X\] percentage points (95% CI \[X\] to \[X\]). DiD estimated savings of $\[X\] PMPM; the naive before/after estimate of $\[X\] overstated savings by \[X%\] due to regression to the mean.

## Project 6 — Clinical NLP and healthcare data standards

**Goal:** Extract structured facts from free-text clinical notes — conditions, medications, and social determinants of health (SDOH) — and show how they fill gaps that coded data misses. Then map the output to FHIR so it fits an HIE pipeline.

**Business question:** How many patients have documented housing instability, food insecurity, or uncontrolled conditions in notes but no corresponding code, and does that change their risk score?

### Pipeline

1. **Section detection:** split notes into History, Assessment, Plan, Social History; SDOH lives mostly in Social History.
2. **Entity extraction:** rule-based target matching with medspaCy for SDOH phrases, plus scispaCy or a clinical transformer (for example, Bio\_ClinicalBERT fine-tuned on a labeled sample) for conditions.
3. **Context:** negation (“denies homelessness”), historical (“history of”), hypothetical (“if symptoms recur”), and family history (“mother had breast cancer”) are filtered out with the ConText algorithm.
4. **Normalization:** map conditions to SNOMED CT / ICD-10 and SDOH findings to ICD-10 Z codes (Z59.0 homelessness, Z59.41 food insecurity).
5. **Evaluation:** two-annotator labeled gold set of \[N\] notes; report precision, recall, F1 and inter-annotator agreement (Cohen’s kappa).

```python
import medspacy
from medspacy.ner import TargetRule

nlp = medspacy.load()   # includes sentence splitting, ConText, sectionizer
rules = [
    TargetRule("housing instability", "SDOH_HOUSING",
               pattern=[{"LOWER": {"IN": ["homeless", "homelessness", "evicted", "shelter"]}}]),
    TargetRule("food insecurity", "SDOH_FOOD",
               pattern=[{"LOWER": "food"}, {"LOWER": {"IN": ["insecurity", "bank", "stamps"]}}]),
]
nlp.get_pipe("medspacy_target_matcher").add(rules)

def extract(note_text):
    doc = nlp(note_text)
    return [(e.label_, e.text) for e in doc.ents
            if not (e._.is_negated or e._.is_historical or e._.is_family or e._.is_hypothetical)]

extract("Pt denies homelessness. Currently staying in a shelter. Uses food bank weekly.")
# -> [('SDOH_HOUSING', 'shelter'), ('SDOH_FOOD', 'food bank')]
```

### Standards layer

| Standard | Where it appears in this project |
| --- | --- |
| HL7 v2 (ADT, ORU) | Source format for admissions and lab results; parse with `hl7apy` |
| FHIR R4 | Output: each SDOH finding becomes an `Observation` linked to the `Patient`; Synthea data is natively FHIR |
| X12 837 / 835 | Claims and remittance structure behind the DE-SynPUF tables |
| ICD-10-CM, CPT/HCPCS, LOINC, RxNorm, SNOMED CT | Code systems used for normalization and measure value sets |
| USCDI | Checklist of required data elements for the completeness checks in Project 1 |

### Advanced add-ons

- **LLM-assisted extraction with guardrails:** compare a prompted large language model against the rules baseline on the same gold set, run only on de-identified text in an approved environment, and report where it helps (paraphrased SDOH) and where it hallucinates.
- **Impact analysis:** add the NLP-derived SDOH flags to the Project 3 readmission model and report the change in AUPRC and the fairness metrics.

### Result template

The pipeline reached F1 \[X\] for housing and \[X\] for food insecurity. \[X%\] of patients with an SDOH finding in notes had no Z code, and adding the flags improved readmission AUPRC from \[X\] to \[X\].

## Project 7 — Executive and clinical dashboards (Tableau and Power BI)

**Goal:** Turn Projects 1–5 into decision tools for three audiences. Build the executive view in Power BI and the clinical view in Tableau Public, so both tools in the posting are demonstrated and one is publicly linkable.

### Dashboard design

| Page | Audience | Decision it supports | Key visuals |
| --- | --- | --- | --- |
| Executive summary | CEO, CDO, leadership | Are cost and quality on track this quarter? | KPI cards with target and trend arrow (TCOC PMPM, readmission O/E, Star estimate), 13-month trend lines |
| Utilization & cost | Finance, network management | Which facilities and service lines drive cost? | Cost-by-setting waterfall, facility funnel plot of readmission rates with control limits |
| Quality & care gaps | Medical directors, clinics | Which patients and measures need action? | Measure rate vs Star cut-points, p-charts, drill-through to a de-identified gap list |
| Risk & outreach | Care managers | Who do we call this week and why? | Risk-tier distribution, ranked list with SHAP top-3 reasons |
| Data quality | Data operations, contributors | Which feeds need fixing? | Contributor scorecard heatmap (dimension × contributor), anomaly timeline |

### Power BI DAX examples

```dax
Readmit Rate 30d =
DIVIDE(
    CALCULATE(COUNTROWS(Stays), Stays[Readmit30d] = 1),
    COUNTROWS(Stays)
)

Readmit O/E =
DIVIDE([Readmit Rate 30d], AVERAGE(Stays[ExpectedReadmitProb]))

PMPM =
DIVIDE(SUM(Claims[PaidAmt]), SUM(MemberMonths[MemberMonths]))

PMPM YoY % =
VAR PriorYear = CALCULATE([PMPM], SAMEPERIODLASTYEAR('Date'[Date]))
RETURN DIVIDE([PMPM] - PriorYear, PriorYear)

-- small-cell suppression: blank any rate built on fewer than 11 patients
Readmit Rate (Safe) =
IF(COUNTROWS(Stays) < 11, BLANK(), [Readmit Rate 30d])
```

### Tableau calculated fields

```text
// Funnel plot control limits for facility readmission rate
[Upper 95% Limit] = [Network Rate] + 1.96 * SQRT([Network Rate] * (1 - [Network Rate]) / [Index Stays])
[Outlier Flag]    = IF [Facility Rate] > [Upper 95% Limit] THEN "Above expected" ELSE "Within range" END
```

### Advanced practices to show

- **Row-level security:** each clinic sees only its own patients; leadership sees aggregates. Demonstrate with Power BI RLS roles.
- **Performance:** star-schema data model (fact tables for claims and stays, dimensions for member, provider, date), aggregations pre-computed in SQL, incremental refresh.
- **Accessibility:** colorblind-safe palette, no meaning carried by color alone, alt text on visuals.
- **Plain-language annotations:** every page has a one-sentence “what this means” headline that updates with the data.
- **Usage feedback loop:** a short release note and a 15-minute walkthrough recording for each audience.

### Result template

Published \[link to Tableau Public\]. The funnel plot identified \[N\] facilities with readmission rates above the 95% limit after risk adjustment; the executive page reduced the monthly reporting packet from \[N\] slides to one interactive view.

## Governance, HIPAA compliance and documentation pack

This section proves the “maintain absolute confidentiality” and “author technical documentation, SOPs and training materials” parts of the posting. Put each item in a `/docs` folder in the repository.

### HIPAA practices built into every project

- **Minimum necessary:** analysis tables carry only the fields a question needs; names, addresses and full dates stay in a restricted layer, and analysts work from a tokenized member key.
- **De-identification:** a reusable Python module applying the HIPAA Safe Harbor method — remove the 18 identifiers, generalize dates to year, truncate ZIP to 3 digits (and set to 000 where the 3-digit area has fewer than 20,000 people), and cap ages above 89 at “90+”. Note when Expert Determination would be the better path.
- **Small-cell suppression:** no published count or rate based on fewer than 11 individuals, matching the CMS cell-size policy.
- **Access and audit:** role-based access, encrypted storage, and a query log; describe how you would respond to a contributor’s request about who accessed their data.
- **42 CFR Part 2 awareness:** substance-use-disorder treatment records need extra consent handling — flag them in the data model rather than mixing them into general analytics.

```python
import pandas as pd

LOW_POP_ZIP3 = {"036", "059", "102", "203", "556", "692", "790", "821", "823", "878", "879", "884", "893"}
# load the current list from Census data rather than hard-coding it in production

def safe_harbor(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=["name", "street", "phone", "email", "ssn", "mrn"], errors="ignore").copy()
    out["zip3"] = out.pop("zip").str[:3].where(lambda z: ~z.isin(LOW_POP_ZIP3), "000")
    for col in ["dob", "admit_dt", "discharge_dt"]:
        if col in out:
            out[col + "_year"] = pd.to_datetime(out.pop(col)).dt.year
    out["age"] = out["age"].clip(upper=90).astype(str).replace("90", "90+")
    return out
```

### Documentation samples

1. **Data dictionary:** for every mart table, list field name, definition, type, source system, transformation rule, allowed values, and PHI classification.
2. **Report specification:** purpose, audience, measure definitions with numerator/denominator, data sources and refresh cadence, exclusions, suppression rules, owner, and sign-off.
3. **Standard operating procedure — handling a data-quality anomaly:** detect (automated alert) → triage within 1 business day → confirm with sample records in the secure environment → notify contributor with record IDs only → track fix → re-validate → close and document root cause.
4. **Training one-pager:** “How to read the readmission funnel plot” for clinic managers, written at an 8th-grade reading level.

### Stakeholder translation example

Show the same finding written twice:

- **For the analytics team:** “Facility 214’s risk-adjusted readmission O/E is 1.31 (95% CI 1.12–1.52) over 4 quarters; the excess concentrates in heart failure discharges to home without a follow-up visit within 7 days.”
- **For a medical director:** “Facility 214 readmits about 3 in 10 more patients than expected for how sick they are. Most of the gap is heart failure patients sent home without a follow-up appointment in the first week. Scheduling that visit before discharge is the most likely fix; we can track it monthly on the dashboard.”

## How to build and present this portfolio

Build Projects 1, 2, 3 and 7 first — they cover every required qualification — then add 4, 5 and 6 as time allows. A realistic pace is one project per week.

### Repository structure

```text
uhin-health-analytics-portfolio/
├── README.md                 # positioning statement, project table, links to dashboards
├── data/                     # download scripts only; never commit data files
├── sql/
│   ├── staging/              # raw → cleaned tables
│   ├── marts/                # stays, member_months, measures, risk scores
│   └── tests/                # dbt or SQL assertions
├── notebooks/                # one per project, each opens with the business question
├── src/                      # reusable Python: dq_checks.py, deid.py, hcc.py, nlp.py
├── dashboards/               # .pbix, Tableau Public links, screenshots
├── docs/                     # data dictionary, report spec, SOP, model card
└── environment.yml
```

Each project README should follow the same five headings: business question, data, method, result (with your real numbers), and what I would do next. Recruiters skim; that pattern lets them find the answer in 30 seconds.

### Interview talking points

- **Data quality story (Project 1):** I describe one anomaly I caught, how I prove it, and how I should tell the contributor without exposing PHI. UHIN runs an HIE, so this is likely their daily reality.
- **The “why not before/after” story (Project 5):** I explain regression to the mean in two sentences in the style of statistical maturity.
- **Calibration over accuracy (Project 3):** I explain why a care manager needs an honest probability, not just a high AUROC.
- **Claims plus clinical (Project 4):** the rate gain from adding clinical data is the business case for an HIE — say it in UHIN’s terms.
- **Translation (governance pack):** to read the two versions of the same finding aloud.

### 30-60-90 day plan to bring to the final interview

1. **Days 1–30 — learn:** complete HIPAA and UHIN privacy training; map the CHIE and claims data sources, owners and refresh schedules; meet the CDO and key stakeholders; reproduce two existing reports to learn the data.
2. **Days 31–60 — stabilize:** stand up automated quality checks on the highest-volume feeds; document data dictionaries for the most-used tables; deliver one quick-win dashboard improvement.
3. **Days 61–90 — extend:** deliver a first risk-stratified or quality-measure dataset for the CDO’s predictive work; publish a contributor data-quality scorecard; propose one evaluation study for a value-based care initiative.

### Final checklist before to share the link

- [ ] Every \[bracket\] replaced with a number I produced
- [ ] No real patient data anywhere in the repository or screenshots
- [ ] Dashboards load from a public link without a login
- [ ] Each notebook runs top to bottom from a clean environment



