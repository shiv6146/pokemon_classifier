from __future__ import print_function
from keras.models import load_model
from scipy import misc
import numpy as np
import imageio
import json
import train

def process_input(inp):
    try:
        return np.array(misc.imresize(imageio.imread(inp, pilmode="L"), (32, 32)).flatten()).reshape(1, 1024)
    except Exception as e:
        return None
    

# Get datasets
# img_data, label_data = train.load_data()

# Train and save the model
# train.train_and_save(img_data, label_data)

# Load trained model
model = load_model('pokemon_classifier.h5')

x = raw_input("Enter an image URL of any above listed pokemon: ")
x = process_input(x)
if x is not None:
    prediction = model.predict_classes(x)
    with open("mappings.json") as fp:
        labels = json.load(fp)
        print("Umm! You must be a ", labels[str(prediction[0])])
else:
    print("Oops! Unable to parse the URL/filepath")
