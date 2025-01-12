# DeepSAD + Fairness Loss for AKI Prediction

> **Note:** As per the guidelines from the organizing team at Seoul National University Bundang Hospital, sharing any information related to the actual dataset is prohibited. Therefore, only the model code used in this project is provided here.

---

## **Fairness Metrics**
We employed **Demographic Parity (DP)** and **Equalized Odds (EO)** as fairness metrics:  
- **DP (Demographic Parity):** The **predicted positive rate** should be similar between the protected group and the non-protected group.  
- **EO (Equalized Odds):** For true positives/negatives, the **predicted positive rate** should be similar across groups.

### **Final Performance**
- **AUC:** 88.21%

---

## **Execution Instructions**

To run the model, specify the fairness type using the `--fairness_type` argument:  
Options: `[EO, DP]`

### Example Command:
```bash
python main.py custom custom_mlp ../log/DeepSAD/AKI_test ../data/ \
  --fairness_type EO \
  --ratio_known_normal 0.6 \
  --ratio_known_outlier 0.4 \
  --ratio_pollution 0.0 \
  --lr 0.001 \
  --n_epochs 10 \
  --lr_milestone 50 \
  --batch_size 128 \
  --weight_decay 0.5e-6 \
  --pretrain True \
  --ae_lr 0.001 \
  --ae_n_epochs 10 \
  --ae_batch_size 128 \
  --ae_weight_decay 0.5e-3 \
  --normal_class 0 \
  --known_outlier_class 1 \
  --n_known_outlier_classes 1 \
  --seed 0
```
