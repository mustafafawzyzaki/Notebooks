# Obesity Classification – Model Documentation

## Model Overview

| Property | Value |
|---|---|
| **Algorithm** | Random Forest (300 trees, balanced class weights) |
| **Best Accuracy** | 92.82 % |
| **Best Macro F1** | 0.9103 |
| **Best Weighted F1** | 0.9286 |
| **Saved File** | `obesity_rf_model.joblib` (pipeline + target encoder) |

---

## Input Features

### Numerical Features (7)

| Abbreviation | Full Name | Description | Range / Unit |
|---|---|---|---|
| **Age** | Age | Person's age in years | float (e.g. `24.0`) |
| **FCVC** | Frequency of Consumption of Vegetables | How often vegetables are consumed | 1 (never) → 3 (always) |
| **NCP** | Number of Main Meals per Day | Number of main meals eaten daily | 1 – 4 |
| **CH2O** | Consumption of Water Daily | Daily water intake level | 1 (< 1 L) → 3 (> 2 L) |
| **FAF** | Frequency of Physical Activity | How often physical activity is performed | 0 (none) → 3 (4–5 days/week) |
| **TUE** | Time Using Technology Devices | Daily hours spent on phones, computers, video games, etc. | 0 – 2 |
| **BMI** | Body Mass Index | Calculated as **Weight (kg) ÷ Height² (m²)** | float (e.g. `28.3`) |

> **Note:** `Height` and `Weight` are **not** direct model inputs. The user must compute `BMI = Weight / Height²` and supply that value. BMI is the single most important feature (≈ 59 % of model importance).

### Categorical Features (8)

| Abbreviation | Full Name | Description | Possible Values |
|---|---|---|---|
| **Gender** | Gender | Person's gender | `Male`, `Female` |
| **family_history_with_overweight** | Family History with Overweight | Whether a family member has suffered or suffers from overweight | `yes`, `no` |
| **FAVC** | Frequent Consumption of High Caloric Food | Whether the person frequently eats high-calorie food | `yes`, `no` |
| **CAEC** | Consumption of Food Between Meals | How often food is eaten between main meals | `no`, `Sometimes`, `Frequently`, `Always` |
| **SMOKE** | Smoking | Whether the person smokes | `yes`, `no` |
| **SCC** | Calories Consumption Monitoring | Whether the person monitors their calorie intake | `yes`, `no` |
| **CALC** | Consumption of Alcohol | How often alcohol is consumed | `no`, `Sometimes`, `Frequently`, `Always` |
| **MTRANS** | Main Mode of Transportation | The primary transportation method used | `Automobile`, `Bike`, `Motorbike`, `Public_Transportation`, `Walking` |

---

## Output (Prediction Result)

The model returns **one of four obesity-level classes**:

| Class | Original Labels (before grouping) | Typical BMI Range |
|---|---|---|
| **Underweight** | Insufficient_Weight | < 18.5 |
| **Normal** | Normal_Weight | 18.5 – 24.9 |
| **Overweight** | Overweight_Level_I, Overweight_Level_II | 25.0 – 29.9 |
| **Obesity** | Obesity_Type_I, Obesity_Type_II, Obesity_Type_III | ≥ 30.0 |

---

## Feature Importance (Random Forest)

Ranked by model-based feature importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | BMI | 0.5915 |
| 2 | family_history_with_overweight | 0.0299 |
| 3 | Age | 0.0177 |
| 4 | CH2O | 0.0101 |
| 5 | Gender | 0.0090 |
| 6 | TUE | 0.0055 |
| 7 | FAF | 0.0054 |
| 8 | NCP | 0.0053 |
| 9 | CALC | 0.0050 |
| 10 | CAEC | 0.0049 |
| 11 | MTRANS | 0.0030 |
| 12 | FCVC | 0.0028 |
| 13 | FAVC | 0.0027 |
| 14 | SMOKE | 0.0001 |
| 15 | SCC | 0.0001 |

---

## Example Prediction Call

```python
result = predict_obesity(
    art,
    Age=24.0,                                   # Age in years
    FCVC=2.0,                                    # Frequency of Consumption of Vegetables (1-3)
    NCP=3.0,                                     # Number of Main Meals per Day (1-4)
    CH2O=2.8,                                    # Consumption of Water Daily (1-3)
    FAF=0.0,                                     # Frequency of Physical Activity (0-3)
    TUE=1.0,                                     # Time Using Technology Devices (0-2)
    BMI=28.3,                                    # Body Mass Index = Weight / Height²
    Gender="Male",                               # Gender
    family_history_with_overweight="yes",         # Family History with Overweight
    FAVC="yes",                                  # Frequent Consumption of High Caloric Food
    CAEC="Sometimes",                            # Consumption of Food Between Meals
    SMOKE="no",                                  # Smoking
    SCC="no",                                    # Calories Consumption Monitoring
    CALC="Sometimes",                            # Consumption of Alcohol
    MTRANS="Public_Transportation",              # Main Mode of Transportation
)
print(f"Predicted class: {result}")
# Output → Predicted class: Overweight
```

---

## Important Notes

- **BMI dominates** the model because obesity categories are directly derived from BMI-based clinical definitions. This makes the classifier a *BMI-assisted obesity classifier*, not a pure lifestyle-only predictor.
- If the goal is to study **lifestyle predictors only**, remove BMI (and Height/Weight) and train a separate model.
- **Class weights** (`balanced`) were used during training to reduce the effect of class imbalance.
- **Permutation importance** was also computed to validate feature ranking, since summing one-hot encoded columns can inflate categorical feature importance.
