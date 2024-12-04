import unittest
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift

class TestMnistMeanShift(unittest.TestCase):
    def test_scaling(self):
        # Test skalowania danych
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        self.assertAlmostEqual(np.mean(scaled_data), 0, delta=1e-7)
        self.assertAlmostEqual(np.std(scaled_data), 1, delta=1e-7)

    def test_pca_reduction(self):
        # Test redukcji wymiarów PCA
        data = np.random.rand(100, 10)
        pca = PCA(n_components=5)
        reduced_data = pca.fit_transform(data)
        self.assertEqual(reduced_data.shape[1], 5)

    def test_mean_shift(self):
        # Test klasteryzacji Mean Shift
        data, _ = make_classification(
            n_samples=100,
            n_features=6,
            n_classes=3,
            n_redundant=1,
            n_repeated=1,
            n_informative=3,  # Liczba informatywnych cech mniejsza niż n_features
            random_state=42
        )
        mean_shift = MeanShift(bandwidth=2)
        mean_shift.fit(data)
        clusters = mean_shift.labels_
        self.assertTrue(len(np.unique(clusters)) > 1)

    def test_mode_mapping(self):
        # Test mapowania klastrów do cyfr
        y_true = np.array([0, 1, 1, 2, 2, 2])
        cluster_labels = np.array([0, 1, 1, 2, 2, 2])
        cluster_to_digit = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster_label in unique_clusters:
            mask = (cluster_labels == cluster_label)
            most_common = mode(y_true[mask], axis=None)
            cluster_to_digit[cluster_label] = most_common.mode.item()

        mapped_clusters = np.array([cluster_to_digit[label] for label in cluster_labels])
        self.assertTrue(np.array_equal(y_true, mapped_clusters))

    def test_confusion_matrix(self):
        # Test macierzy pomyłek
        y_true = [0, 1, 2, 2, 1, 0]
        y_pred = [0, 1, 2, 1, 1, 0]
        conf_matrix = confusion_matrix(y_true, y_pred)
        self.assertEqual(conf_matrix.shape, (3, 3))
        self.assertEqual(np.sum(conf_matrix), len(y_true))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
