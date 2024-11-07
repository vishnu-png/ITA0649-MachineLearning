import pandas as pd

def find_s_algorithm(data, target_column):
    target = data[target_column]
    features = data.drop(columns=[target_column])
    
    specific_hypothesis = list(features.iloc[0])  
    for i, example in data.iterrows():
        example_features = example.drop(target_column)
        example_target = target[i]
        
        if example_target == 'Yes':
            for j, value in enumerate(example_features):
                if specific_hypothesis[j] != value:
                    specific_hypothesis[j] = '?'  
    
    print("Most Specific Hypothesis:", specific_hypothesis)


if __name__ == "__main__":
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Overcast'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Medium', 'Low'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes']
    }

    df = pd.DataFrame(data)
    find_s_algorithm(df, 'PlayTennis')
