from sklearn.mixture import GaussianMixture
import pandas as pd

data = pd.DataFrame({
    'Feature1': [5, 10, 15, 10, 5, 10, 15, 10],
    'Feature2': [10, 20, 10, 20, 10, 20, 10, 20]
})

gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data)

labels = gmm.predict(data)
print("Estimated labels:", labels)
print("Means of each component:\n", gmm.means_)
