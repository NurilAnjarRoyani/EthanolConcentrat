import pandas as pd
import numpy as np

# Contoh data time series 1751 x 3 (sederhana)
df_input = pd.DataFrame({
    "Channel 1": np.random.rand(1751),
    "Channel 2": np.random.rand(1751),
    "Channel 3": np.random.rand(1751)
})

# Misal prediksi kelas & probabilitas (dummy)
logreg_classes = ['E30','E35','E40','E45']
pred_class = "E35"
pred_prob = np.array([0.08, 0.26, 0.59, 0.07])  # contoh probabilitas

# DataFrame probabilitas
prob_df = pd.DataFrame([pred_prob], columns=logreg_classes)
prob_df_sorted = prob_df.T.sort_values(by=0, ascending=False).rename(columns={0:"Probabilitas"})

# Statistik dasar
stats = pd.DataFrame({
    "Mean": df_input.mean(),
    "Std": df_input.std(),
    "Min": df_input.min(),
    "Max": df_input.max(),
    "Median": df_input.median()
}, index=["Channel 1", "Channel 2", "Channel 3"])

# Tampilkan hasil
print("=== Hasil Prediksi ===")
print(f"Prediksi kelas Ethanol: {pred_class}\n")
print("=== Probabilitas Prediksi per Kelas (Terurut) ===")
print(prob_df_sorted, "\n")
print("=== Statistik Dasar Time Series ===")
print(stats)
