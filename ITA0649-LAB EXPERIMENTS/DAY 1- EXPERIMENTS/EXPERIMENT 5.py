import numpy as np
import pandas as pd
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_nearest_neighbors(train_data, test_point, k):
    distances = []
    
    for index, train_point in train_data.iterrows():
        train_features = train_point.drop('Class')
        distance = euclidean_distance(test_point, train_features)
        distances.append((distance, train_point['Class']))
    
    distances.sort(key=lambda x: x[0])
    
    neighbors = distances[:k]
    
    neighbor_classes = [neighbor[1] for neighbor in neighbors]
    
    most_common_class = Counter(neighbor_classes).most_common(1)[0][0]
    
    return most_common_class

if __name__ == "__main__":
    data = {
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [1, 2, 3, 4, 5],
        'Class': ['A', 'A', 'B', 'B', 'A']
    }

    df = pd.DataFrame(data)

    X_train = df.drop(columns=['Class'])
    y_train = df['Class']

    test_point = np.array([3, 3])

    k = 3

    predicted_class = k_nearest_neighbors(df, test_point, k)
    
    print(f'The predicted class for the test point {test_point} is: {predicted_class}')
