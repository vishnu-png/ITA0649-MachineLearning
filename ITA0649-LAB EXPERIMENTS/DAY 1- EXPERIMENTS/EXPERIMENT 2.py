import pandas as pd

def candidate_elimination_algorithm(data, target_column):
    target = data[target_column]
    features = data.drop(columns=[target_column])

    specific_hypothesis = ['?'] * len(features.columns)  
    general_hypothesis = [['?' for _ in range(len(features.columns))] for _ in range(len(features.columns))]  

    for i, example in data.iterrows():
        example_features = example.drop(target_column)
        example_target = target[i]

        if example_target == 'Yes': 
            for j, value in enumerate(example_features):
                if specific_hypothesis[j] == '?':
                    specific_hypothesis[j] = value
                elif specific_hypothesis[j] != value:
                    specific_hypothesis[j] = '?'  
            for j, value in enumerate(example_features):
                if general_hypothesis[j] != '?' and general_hypothesis[j] != value:
                    general_hypothesis[j] = '?'
        else:  
            for j, value in enumerate(example_features):
                if general_hypothesis[j] != '?' and general_hypothesis[j] != value:
                    general_hypothesis[j] = value

    print("Final Specific Hypothesis:", specific_hypothesis)
    print("Final General Hypotheses:", general_hypothesis)


if __name__ == "__main__":
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Overcast'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Medium', 'Low'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes']
    }

    df = pd.DataFrame(data)

    candidate_elimination_algorithm(df, 'PlayTennis')
