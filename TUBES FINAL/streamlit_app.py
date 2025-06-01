import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# =====================================
# Load semua model & resource
# =====================================
logreg_model = joblib.load('model_kanker.pkl')
nb_model = joblib.load('model_kanker_nb.pkl')

scaler = joblib.load('scaler_kanker.pkl')
gender_encoder = joblib.load('label_encoder_Gender.pkl')
country_encoder = joblib.load('label_encoder_Country_Region.pkl')
cancer_type_encoder = joblib.load('label_encoder_Cancer_Type.pkl')
cancer_stage_encoder = joblib.load('label_encoder_Cancer_Stage.pkl')

X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')
y_prob_logreg = joblib.load('y_prob_logreg.pkl')
y_prob_nb = joblib.load('y_prob_nb.pkl')
df_scaled = joblib.load('df_scaled.pkl')

# =====================================
# Konfigurasi Halaman
# =====================================
st.set_page_config(page_title="Prediksi Keparahan Kanker", layout="centered")
st.title("🎗️ Prediksi Tingkat Keparahan Kanker")

st.markdown("Aplikasi ini memprediksi tingkat keparahan kanker menggunakan data pasien dan model machine learning.")
st.markdown("---")

# =====================================
# Pilih Model
# =====================================
model_choice = st.selectbox("Pilih Model yang Ingin Digunakan", ["Logistic Regression", "Naive Bayes"])
model = logreg_model if model_choice == "Logistic Regression" else nb_model

st.subheader(f"📊 Evaluasi Model ({model_choice})")

X_test_scaled = scaler.transform(X_test)
y_pred_test = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("Recall", f"{recall:.4f}")
with col2:
    st.metric("Precision", f"{precision:.4f}")
    st.metric("F1 Score", f"{f1:.4f}")

# Confusion Matrix
st.markdown("#### Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3], yticklabels=[1, 2, 3], ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
st.pyplot(fig)

# ROC Curve
st.markdown("#### ROC Curve")
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

selected_prob = y_prob_logreg if model_choice == "Logistic Regression" else y_prob_nb

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], selected_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
colors = ['blue', 'green', 'red', 'orange', 'purple']

for i in range(n_classes):
    ax_roc.plot(fpr[i], tpr[i], color=colors[i % len(colors)],
                label=f'Class {i+1} (AUC = {roc_auc[i]:.2f})')

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title(f'ROC Curve - {model_choice}')
ax_roc.legend(loc='lower right')
ax_roc.grid(True)
st.pyplot(fig_roc)

st.markdown("---")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df_scaled, palette="Set2", ax=ax)
ax.set_title("Visualisasi Klaster dengan PCA")
st.pyplot(fig)

# =====================================
# Form Input Data Pasien
# =====================================
st.subheader("🧾 Input Data Pasien")

age = st.slider("Umur", 0, 100, 30)
gender = st.selectbox("Jenis Kelamin", gender_encoder.classes_)
country = st.selectbox("Negara", country_encoder.classes_)
year = st.slider("Tahun Diagnosis", 2000, 2025, 2021)

genetic_risk = st.slider("Genetic Risk (0-10)", 0.0, 10.0, 5.0)
air_pollution = st.slider("Air Pollution (0-10)", 0.0, 10.0, 5.0)
alcohol_use = st.slider("Alcohol Use (0-10)", 0.0, 10.0, 5.0)
smoking = st.slider("Smoking (0-10)", 0.0, 10.0, 5.0)
obesity_level = st.slider("Obesity Level (0-10)", 0.0, 10.0, 5.0)

cancer_type = st.selectbox("Jenis Kanker", cancer_type_encoder.classes_)
cancer_stage = st.selectbox("Stadium Kanker", cancer_stage_encoder.classes_)

treatment_cost = st.number_input("Biaya Pengobatan (USD)", value=10000.0)
survival_years = st.slider("Perkiraan Tahun Bertahan Hidup", 0.0, 20.0, 5.0)

# =====================================
# Prediksi
# =====================================
if st.button("🔍 Prediksi Tingkat Keparahan"):
    input_df = pd.DataFrame([[
        age,
        gender_encoder.transform([gender])[0],
        country_encoder.transform([country])[0],
        year,
        genetic_risk,
        air_pollution,
        alcohol_use,
        smoking,
        obesity_level,
        cancer_type_encoder.transform([cancer_type])[0],
        cancer_stage_encoder.transform([cancer_stage])[0],
        treatment_cost,
        survival_years
    ]], columns=X_test.columns)

    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    severity_map = {1: "Ringan", 2: "Sedang", 3: "Berat"}
    st.success(f"🎯 **Tingkat Keparahan Kanker:** {severity_map.get(pred, 'Tidak Diketahui')}")
