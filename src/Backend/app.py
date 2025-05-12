from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model

app = Flask(__name__)
CORS(app)  

model = load_model('lstm.h5')
save_model(model, 'lstm_retrained.h5')

@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.get_json()

    if "input" not in data:
        return jsonify({"error": "Invalid input, 'input' key is missing"}), 400

    input_list = data["input"]
    print("Received input data:", input_list)

    try:
        
        input_data = np.array(input_list).reshape(1, len(input_list), 1)
        print("Reshaped input data:", input_data.shape)

        prediction = model.predict(input_data)
        print("Prediction:", prediction)

        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)