# MIMIC-IV ICU Preprocessing Files

## caregiver
- **Description**: Contains information about the healthcare providers who took care of the patients.
- **Usage**: Although it provides caregiver-related metadata, it is not directly used for AKI prediction in our current workflow.

---

## chartevents
- **Description**: This is one of the most critical files for AKI prediction. It records patients’ vital signs and other clinical measurements (e.g., blood pressure, heart rate, oxygen saturation) in a time-series format.
- **Preprocessing Notebook**: **`chartevents.ipynb`**  
  - Handles data cleaning, feature selection, and time-series alignment for AKI prediction tasks.

---

## d_items
- **Description**: Provides descriptive information about the items (variables) used in `chartevents.csv.gz`.
- **Usage**: Primarily utilized to map item IDs to meaningful labels in `chartevents`, aiding in feature engineering.

---

## datetimeevents
- **Description**: Contains date- and time-related events.
- **Usage**: Not used in our current AKI prediction pipeline because it did not provide significant clinical insights for this task.

---

## icustays
- **Description**: Contains ICU admission records (admission time, discharge time, ward information, etc.).
- **Usage**: Used for filtering ICU-specific admissions and aligning time windows for AKI-related events.

---

## ingredientevents
- **Description**: Logs events related to the components of administered medications.
- **Preprocessing Notebook**: **`Ingredients.ipynb`**  
  - Used to extract relevant medication components that could be indicative of or correlated with AKI.

---

## inputevents
- **Description**: Records input events such as fluids and medications administered to patients.
- **Preprocessing Notebook**: **`inputevents.ipynb`**  
  - Plays a crucial role in understanding medication/fluids that may affect kidney function.

---

## outputevents
- **Description**: Tracks output events such as urine output, drainage volume, etc.
- **Preprocessing Notebook**: **`outputevents.ipynb`**  
  - Especially relevant for AKI prediction since urine output is a key indicator of renal function.

---

## procedureevents
- **Description**: Contains information about various procedures performed in the ICU.
- **Preprocessing Notebook**: **`procedure.ipynb`**  
  - Relevant procedures (e.g., dialysis) are critical for understanding kidney-related interventions.

---

## Other MIMIC-IV Datasets
- **Note**: Files or tables not listed here were excluded because they did not provide meaningful information for predicting AKI in our specific use case.

