# ---------------- MODEL 1 ------------------
# testing import
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("/content/drive/MyDrive/UIGM kuliah /Jurnal/PemMesin/Salinan klasifikasimhs (1).csv")  # ganti dengan nama file CSV kamu
print(data.head())
# Menghapus data kosong (jika ada)
data.dropna(inplace=True)

# Label encoding (contoh jika ada kolom 'pekerjaan_ortu')
data = pd.get_dummies(data, columns=['Pekerjaan Orang Tua'])

# Pisahkan fitur dan target
X = data.drop('Kelayakan Keringanan UKT', axis=1)  # kolom 'kelayakan' = label
y = data['Kelayakan Keringanan UKT']

# 5. Normalisasi kolom numerik dengan rentang 0–1
numerical_cols = ['Penghasilan Orang Tua', 'Kendaraan']
scaler = MinMaxScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Inisialisasi dan latih model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Prediksi data uji
y_pred = model.predict(X_test)

# 9. Evaluasi model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
