import math
import pandas as pd

def entropy(data):
    value_counts = data.iloc[:, -1].value_counts(normalize=True)
    return -sum(value_counts * value_counts.apply(lambda x: math.log2(x) if x > 0 else 0))

def information_gain(data, attribute):
    total_entropy = entropy(data)

    values = data[attribute].unique()
    
    weighted_entropy = 0
    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    
    return total_entropy - weighted_entropy

def id3(data, attributes, target_attribute):
    if len(data[target_attribute].unique()) == 1:
        return data[target_attribute].iloc[0]
    
    if len(attributes) == 0:
        return data[target_attribute].mode()[0]
    
    gains = {attribute: information_gain(data, attribute) for attribute in attributes}
    
    best_attribute = max(gains, key=gains.get)
    
    tree = {best_attribute: {}}
    
    remaining_attributes = [attribute for attribute in attributes if attribute != best_attribute]
    
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        tree[best_attribute][value] = id3(subset, remaining_attributes, target_attribute)
    
    return tree


def classify(tree, sample):
    if not isinstance(tree, dict):  
        return tree
    
    attribute = list(tree.keys())[0]  
    attribute_value = sample[attribute]
    
    return classify(tree[attribute][attribute_value], sample)

if __name__ == "__main__":
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Medium', 'Medium', 'Low', 'Low', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    
    df = pd.DataFrame(data)
    
    attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    target_attribute = 'PlayTennis'
    
    decision_tree = id3(df, attributes, target_attribute)
    
    print("Decision Tree:")
    print(decision_tree)
    
    new_sample = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
    prediction = classify(decision_tree, new_sample)
    print(f"Prediction for new sample: {prediction}")
