# Dataton for AKI Prediction in SNUBH

**Period:** 2024.11 SEP ~ 2024.23 OCT

## Dataset
### MIMIC-IV
- **[MIMIC-IV](https://physionet.org/content/mimiciv/2.2/)** is a large-scale, de-identified electronic health records (EHR) dataset, collected from the Beth Israel Deaconess Medical Center in Boston.  
- It includes data from over **500,000 hospital admissions** for **380,000+ unique patients**, and spans the years 2008 to 2019.  
- The dataset provides structured information such as demographics, vital signs, laboratory test results, medications, and clinical notes, as well as information on **more than 60,000 ICU stays**.  
- Due to its comprehensive nature and public availability, MIMIC-IV is widely used in research on clinical data mining and machine learning.

## Evaluation Method
- We evaluate both **model performance** and **fairness metrics**.

## Methodology
1. Treat data from patients with Acute Kidney Injury (AKI) as anomalies.
2. Perform anomaly detection using **DeepSAD**.
3. Consider fairness metrics, primarily **Demographic Parity (DP)** and **Equalized Odds (EO)**, with **gender** as the sensitive attribute.
   - **DP (Demographic Parity):** The predicted positive rate should be similar between the protected group and the non-protected group.
   - **EO (Equalized Odds):** The predicted positive rate should be similar between groups for both actual positive and actual negative cases.

## Preprocessing Folders
1. **ICU Folder**: Preprocessing of clinical data from the ICU system.
2. **Hosp Module**: Preprocessing of EHR (Electronic Health Records) data throughout the hospital.

## Limitation
While we treated AKI as an outlier in the MIMIC-IV dataset and built an anomaly detection model, we recognized that the **complex nature of real clinical data** (e.g., high dimensionality, time-series characteristics, mismatched test intervals) necessitates **further approaches to improve model performance**.
