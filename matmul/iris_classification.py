import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Load dataset Iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# 2. Preprocessing Data
# One-hot encoding untuk label kelas (3 kelas: Setosa, Versicolor, Virginica)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Split data menjadi Data Latih (80%) dan Data Uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Standarisasi fitur agar pembelajaran model optimal
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Membangun Model Keras (Sequential API)
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)), # Input layer (4 fitur)
    keras.layers.Dense(8, activation='relu'),                     # Hidden layer
    keras.layers.Dense(3, activation='softmax')                   # Output layer (3 kelas)
])

# 4. Kompilasi Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Melatih Model
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

# 6. Evaluasi Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi Data Uji: {accuracy * 100:.2f}%")

# 7. Contoh Prediksi Data Baru
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]]) # Fitur: sepal length, sepal width, petal length, petal width
sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled, verbose=0)
predicted_class = iris.target_names[np.argmax(prediction)]

print(f"Hasil Prediksi Jenis Iris: {predicted_class}")