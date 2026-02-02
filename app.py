from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import register_keras_serializable
from keras.models import load_model
from PIL import Image
import io

# ---------------- CUSTOM LAYER ----------------

@register_keras_serializable(package="Custom")
class HubLayerWrapper(tf.keras.layers.Layer):
    def __init__(self, model_url, input_shape=None, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.model_url = model_url
        self.input_shape = input_shape
        self.trainable = trainable
        self.hub_layer = hub.KerasLayer(
            model_url,
            trainable=trainable,
            input_shape=input_shape
        )

    def call(self, inputs):
        return self.hub_layer(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_url": self.model_url,
            "input_shape": self.input_shape,
            "trainable": self.trainable,
        })
        return config

# ---------------- BREED NAMES ----------------
# Replace this with your full list of 120 breeds in order
breed_names = [
    "Affenpinscher", "Afghan Hound", "African Hunting Dog", "Airedale", "American Staffordshire Terrier",
    "Appenzeller", "Australian Terrier", "Basenji", "Basset", "Beagle", "Bedlington Terrier",
    "Bernese Mountain Dog", "Black-and-tan Coonhound", "Walker Hound", "Bloodhound", "Bluetick",
    "Border Collie", "Border Terrier", "Borzoi", "Boston Bull", "Bouvier Des Flandres", "Boxer",
    "Brabancon Griffon", "Briard", "Brittany Spaniel", "Bull Mastiff", "Cairn", "Cardigan",
    "Chesapeake Bay Retriever", "Chihuahua", "Chow", "Clumber", "Cocker Spaniel", "Collie",
    "Curly-coated Retriever", "Dandie Dinmont", "Dhole", "Dingo", "Doberman", "English Foxhound",
    "English Setter", "English Springer", "Entlebucher", "Eskimo Dog", "Flat-coated Retriever",
    "French Bulldog", "German Shepherd", "German Short-haired Pointer", "Giant Schnauzer",
    "Golden Retriever", "Gordon Setter", "Great Dane", "Great Pyrenees", "Greater Swiss Mountain Dog",
    "Gronendael", "Ibizan Hound", "Irish Setter", "Irish Terrier", "Irish Water Spaniel",
    "Irish Wolfhound", "Italian Greyhound", "Japanese Spaniel", "Keeshond", "Kelpie",
    "Kerry Blue Terrier", "Komondor", "Kuvasz", "Labrador Retriever", "Lakeland Terrier",
    "Leonberg", "Lhasa", "Malamute", "Malinois", "Maltese Dog", "Mexican Hairless",
    "Miniature Pinscher", "Miniature Poodle", "Miniature Schnauzer", "Newfoundland",
    "Norfolk Terrier", "Norwegian Elkhound", "Norwich Terrier", "Old English Sheepdog",
    "Otterhound", "Papillon", "Pekinese", "Pembroke", "Pomeranian", "Pug", "Redbone",
    "Rhodesian Ridgeback", "Rottweiler", "Saint Bernard", "Saluki", "Samoyed", "Schipperke",
    "Scotch Terrier", "Scottish Deerhound", "Sealyham Terrier", "Shetland Sheepdog",
    "Shiba Inu", "Shih-Tzu", "Siberian Husky", "Silky Terrier", "Soft-coated Wheaten Terrier",
    "Staffordshire Bullterrier", "Standard Poodle", "Standard Schnauzer", "Sussex Spaniel",
    "Tibetan Mastiff", "Tibetan Terrier", "Toy Poodle", "Toy Terrier", "Vizsla",
    "Walker Hound", "Weimaraner", "Welsh Springer Spaniel", "West Highland White Terrier",
    "Whippet", "Wire-haired Fox Terrier", "Yorkshire Terrier"
]

# ---------------- LOAD MODEL ----------------

model = load_model(
    "model.keras",
    custom_objects={"HubLayerWrapper": HubLayerWrapper}
)

# ---------------- FLASK APP ----------------

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction_text="No image uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction_text="No image selected")

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        class_index = np.argmax(preds[0])
        confidence = round(float(np.max(preds[0]) * 100), 2)

        # Mapping index to Name
        if class_index < len(breed_names):
            result_text = f"Predicted Breed: {breed_names[class_index]} ({confidence}%)"
        else:
            result_text = f"Predicted Breed: Class {class_index} ({confidence}%)"

        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)