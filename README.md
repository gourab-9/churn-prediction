# 🔍 Customer Churn Prediction App

A **Streamlit web app** for predicting customer churn using a **pretrained machine learning pipeline**. This tool allows business analysts or customer service teams to assess the likelihood of customer churn based on user-provided input.

---

## 🚀 How It Works

This app:

- Loads a pre-trained pipeline (`pipeline.joblib`)
- Accepts user input for all features from the original dataset (`df.pkl`)
- Converts categorical fields to the appropriate format
- Predicts churn using the loaded pipeline
- Displays prediction and input summary

---

## 🗂️ Files

- `app.py` — Streamlit app logic  
- `pipeline.joblib` — Pretrained ML pipeline (e.g., sklearn pipeline)  
- `df.pkl` — Sample dataframe with structure/columns used for input template  

---

## 📦 Installation

Install the required packages:

```bash
pip install streamlit pandas scikit-learn joblib
```

# 🧠 Run the App

To launch the app, open a terminal and run:

```bash
streamlit run app.py
```
✅ Make sure both `pipeline.joblib` and `df.pkl` are in the **same directory** as `app.py`.

---

## 🧾 Example Inputs

The app will prompt the user to enter values for the following types of features:

### Categorical Columns  
➤ Entered as **text inputs** (e.g., `Geography`, `Gender`)

### Binary Fields  
➤ Presented as **Yes/No options**, automatically mapped to 1/0:
- `HasCrCard` → Yes = 1, No = 0  
- `IsActiveMember` → Yes = 1, No = 0

### Numeric Fields  
➤ Entered using **number inputs**, with default values set to the **column median** (from `df.pkl`)

---

## 🧪 Output

After filling in the form and clicking **Predict**, the app will display:

### Churn Prediction  
➤ **Yes** or **No** depending on the model's output

### Input Data  
➤ A **DataFrame-style summary** showing all the values the user entered

---

## ⚠️ Notes

- Ensure that the **column names** and **data types** in `df.pkl` **exactly match** what the trained `pipeline.joblib` expects.
- Binary fields like `HasCrCard` and `IsActiveMember` are **automatically encoded internally** — no manual conversion required.
- **All data preprocessing** (e.g., encoding, scaling, imputation) is handled internally by the pipeline.
- Users only need to **enter raw input values** — the app and model take care of the rest.
