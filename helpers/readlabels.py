import json

def readLabels(model_name):
    # Open the JSON file
    with open('models/'+model_name+'/meta.json', 'r') as f:
        # Load the JSON data into a dictionary
        data = json.load(f)

    # Extract the array from the dictionary
    my_array = data['labels']
    return my_array

