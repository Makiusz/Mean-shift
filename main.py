import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.metrics import confusion_matrix

# Funkcja do wczytania danych MNIST z scikit-learn
def load_mnist():
    print("Wczytywanie danych MNIST...")
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.values  # Obrazy
    y = mnist.target.astype(int).values  # Etykiety
    return X, y

# Funkcja do redukcji wymiarów za pomocą PCA
def pca(X, n_components=2):
    pca_model = PCA(n_components=n_components)
    X_reduced = pca_model.fit_transform(X)
    return X_reduced

# Funkcja jądra Gaussa
def gaussian_kernel(point, points, bandwidth):
    distances = np.linalg.norm(points - point, axis=1)
    weights = np.exp(-distances**2 / (2 * bandwidth**2))
    return weights / np.sum(weights)

# Implementacja Mean Shift
def mean_shift(X, bandwidth, max_iter=300, tol=1e-3):
    centroids = np.copy(X)
    for iteration in range(max_iter):
        new_centroids = []
        for point in centroids:
            weights = gaussian_kernel(point, X, bandwidth)
            weighted_mean = np.average(X, axis=0, weights=weights)
            new_centroids.append(weighted_mean)

        new_centroids = np.array(new_centroids)
        if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
            break
        centroids = new_centroids

    # Przypisz punkty do klastrów
    unique_centroids = []
    labels = []
    for point in centroids:
        for i, centroid in enumerate(unique_centroids):
            if np.linalg.norm(point - centroid) < bandwidth / 2:
                labels.append(i)
                break
        else:
            unique_centroids.append(point)
            labels.append(len(unique_centroids) - 1)

    return np.array(labels), np.array(unique_centroids)

# Funkcja do obliczenia macierzy pomyłek
def confusion_matrix_manual(true_labels, predicted_labels):
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)
    matrix = np.zeros((len(unique_true), len(unique_pred)), dtype=int)

    for true, pred in zip(true_labels, predicted_labels):
        matrix[true, pred] += 1

    return matrix

# Funkcja do dodania liczb wystąpień do macierzy pomyłek
def plot_confusion_matrix_with_counts(cm, ax):
    # Rysowanie macierzy pomyłek
    cax = ax.matshow(cm, cmap="viridis")
    plt.colorbar(cax)

    # Dodawanie liczb wystąpień w każdej komórce
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')

# Główna część programu
if __name__ == "__main__":
    # Wczytywanie danych MNIST
    X, y = load_mnist()

    # Przygotowanie danych (mniejszy zbiór dla testów)
    sample_size = 2000
    X_sample = X[:sample_size]
    y_sample = y[:sample_size]

    # Redukcja wymiarów (PCA)
    print("Redukcja wymiarów (PCA)...")
    X_reduced = pca(X_sample, n_components=2)

    # Klasteryzacja za pomocą Mean Shift (ręcznie)
    print("Klasteryzacja za pomocą Mean Shift (ręcznie)...")
    bandwidth = 10  # Eksperymentalna wartość
    clusters, centroids = mean_shift(X_reduced, bandwidth)

    # Przypisywanie klastrów do rzeczywistych cyfr
    unique_clusters = np.unique(clusters)
    cluster_to_digit = {}
    for cluster_label in unique_clusters:
        mask = clusters == cluster_label
        most_common = mode(y_sample[mask])

        # Upewniamy się, że most_common.mode zawiera wartości
        if hasattr(most_common, 'mode') and most_common.mode.size > 0:
            cluster_to_digit[cluster_label] = most_common.mode.item()  # Pobierz wartość bez indeksowania

    mapped_clusters = np.array([cluster_to_digit[label] for label in clusters])

    # Macierz pomyłek (ręcznie)
    print("Obliczanie macierzy pomyłek (ręcznie)...")
    conf_matrix_manual = confusion_matrix_manual(y_sample, mapped_clusters)

    # Klasteryzacja za pomocą Mean Shift (scikit-learn)
    print("Klasteryzacja za pomocą Mean Shift (scikit-learn)...")
    mean_shift_model = MeanShift(bandwidth=bandwidth)
    mean_shift_model.fit(X_reduced)
    predicted_labels = mean_shift_model.labels_

    # Mapowanie klastrów na cyfry
    unique_clusters_sklearn = np.unique(predicted_labels)
    cluster_to_digit_sklearn = {}
    for cluster_label in unique_clusters_sklearn:
        mask = predicted_labels == cluster_label
        most_common_sklearn = mode(y_sample[mask])

        # Upewniamy się, że most_common_sklearn.mode zawiera wartości
        if hasattr(most_common_sklearn, 'mode') and most_common_sklearn.mode.size > 0:
            cluster_to_digit_sklearn[cluster_label] = most_common_sklearn.mode.item()  # Pobierz wartość bez indeksowania

    mapped_clusters_sklearn = np.array([cluster_to_digit_sklearn[label] for label in predicted_labels])

    # Macierz pomyłek (scikit-learn)
    print("Obliczanie macierzy pomyłek (scikit-learn)...")
    conf_matrix_sklearn = confusion_matrix(y_sample, mapped_clusters_sklearn)

    # Wizualizacja wyników
    print("Wizualizacja wyników...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Macierz pomyłek (ręcznie)
    axes[0].set_title("Macierz pomyłek (ręczna implementacja)")
    axes[0].set_xlabel("Przewidywane")
    axes[0].set_ylabel("Rzeczywiste")
    plot_confusion_matrix_with_counts(conf_matrix_manual, axes[0])

    # Macierz pomyłek (scikit-learn)
    axes[1].set_title("Macierz pomyłek (scikit-learn)")
    axes[1].set_xlabel("Przewidywane")
    axes[1].set_ylabel("Rzeczywiste")
    plot_confusion_matrix_with_counts(conf_matrix_sklearn, axes[1])

    plt.tight_layout()
    plt.show()
