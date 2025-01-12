# MIMIC-IV Hosp Preprocessing Files

Below are other tables available in the MIMIC-IV database along with brief descriptions. Tables not directly relevant to our AKI prediction workflow were excluded from our preprocessing.

---

## omr
- **Description**: Contains day-by-day measurements for inpatients, such as blood pressure (BP), body mass index (BMI), height, weight, and estimated Glomerular Filtration Rate (eGFR).
- **Usage**: Potentially useful for tracking patients’ vital trends over time. Not incorporated into our current AKI model pipeline.

---

## provider
- **Description**: Lists all providers and staff in the database, including possible role identifiers.
- **Usage**: Not used in our AKI workflow as provider data does not directly inform AKI risk.

---

## admission
- **Description**: Provides information regarding a patient’s hospital admissions.
- **Usage**: Helpful for time-based alignment of admission data, though we primarily relied on `icustays.csv.gz` for ICU-specific context.

---

## hcpsevents
- **Description**: Logs requests or events that occurred during a hospital stay.
- **Usage**: Not currently included in our AKI pipeline, as these events were less relevant to kidney function tracking.

---

## diagnoses_icd
- **Description**: Contains all diagnoses assigned to a patient during their hospital stay, using ICD-9 or ICD-10 codes.
- **Usage**: Can be used for comorbidity analysis and risk stratification. However, we did not directly process these codes in our AKI prediction model.

---

## procedures_icd
- **Description**: Lists procedures performed during the hospital stay, encoded with ICD procedure codes.
- **Usage**: Not integrated into the AKI workflow, as other procedure data from `procedureevents.csv.gz` was more granular for our use case.

---

## labevents
- **Description**: Stores laboratory test results, which can be critical for evaluating a patient’s kidney function (e.g., serum creatinine).
- **Usage**: Could provide valuable signals for AKI prediction, but in our project, we focused more on `chartevents.csv.gz` for vital signs. Future iterations may incorporate lab data.

---

## drgcodes
- **Description**: Contains Diagnosis-Related Group (DRG) codes, typically used for hospital reimbursement and categorization of admissions.
- **Usage**: Not used for AKI prediction, as DRG codes do not directly reveal detailed clinical markers of kidney function.

---

## emar
- **Description**: Records medication administrations to patients, often populated by barcode scans performed by nursing staff.
- **Usage**: Not included in the current pipeline. We relied on `inputevents.csv.gz` and `ingredientevents.csv.gz` to capture medication and fluid administration details.

---

## microbiologyevents
- **Description**: Tracks evidence of infections and results of treatments (such as antibiotic susceptibility testing).
- **Usage**: Not part of the current AKI workflow. It may be relevant if AKI is suspected to be related to sepsis or infection in future analyses.

---

## patients
- **Description**: Contains demographic information about the patients.
- **Usage**: We already leveraged demographic data in other files (e.g., `icustays.csv.gz`), so this was not explicitly used in the current AKI pipeline.

---

## services
- **Description**: Describes the medical services provided to the patient during their hospital stay.
- **Usage**: Not directly applicable to AKI prediction in our current approach.

---
