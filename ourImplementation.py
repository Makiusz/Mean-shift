
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

# Wczytywanie danych MNIST
print("Wczytywanie danych MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target.astype(int)

# Przygotowanie danych
print("Przygotowywanie danych...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Zwiększ próbkę dla reprezentatywnych wyników
sample_size = 5000
X_sample = X_scaled[:sample_size]
y_sample = y[:sample_size]

# Redukcja wymiarów
print("Redukcja wymiarów (PCA)...")
pca = PCA(n_components=9)
X_reduced = pca.fit_transform(X_sample)

# Klasteryzacja Mean Shift
print("Klasteryzacja za pomocą Mean Shift...")
bandwidth = estimate_bandwidth(X_reduced, quantile=0.001)
print(f"Oszacowany bandwidth: {bandwidth}")
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(X_reduced)
clusters = mean_shift.labels_

# Sprawdzanie klastrów i przypisywanie cyfr
print("Przypisywanie klastrów do rzeczywistych cyfr...")
unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
print(f"Liczba klastrów: {len(unique_clusters)}")
cluster_to_digit = {}
for cluster_label in unique_clusters:
    mask = (clusters == cluster_label)
    if np.any(mask):  # Sprawdzamy, czy maska nie jest pusta
        most_common = mode(y_sample[mask], axis=None)

        # Obsługa różnej struktury wyniku mode
        if hasattr(most_common, 'mode'):  # Wynik ma atrybut 'mode'
            if np.size(most_common.mode) > 0:  # mode jest tablicą
                cluster_to_digit[cluster_label] = most_common.mode.item()  # Pobierz wartość skalarna
            else:
                cluster_to_digit[cluster_label] = -1  # Klaster pusty, przypisz -1
        else:
            cluster_to_digit[cluster_label] = -1  # Wynik bez atrybutu 'mode', przypisz -1
    else:
        cluster_to_digit[cluster_label] = -1  # Klaster pusty, przypisz -1

# Mapowanie klastrów na cyfry
mapped_clusters = np.array([cluster_to_digit[label] for label in clusters])

# Obliczanie macierzy pomyłek
print("Obliczanie macierzy pomyłek...")
conf_matrix = confusion_matrix(y_sample, mapped_clusters)

# Wizualizacja macierzy pomyłek
print("Wizualizacja wyników...")
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(10))
disp.plot(cmap='viridis', values_format='d')
plt.title("Macierz pomyłek dla Mean Shift na danych MNIST")
plt.show()
