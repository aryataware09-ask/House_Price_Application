import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load the trained linear regression model
with open("linear.pkl", "rb") as f:
  model = pickle.load(f)


@app.route("/")
def home():
  return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
  try:
    # Extract features from form submission
    features = [
        float(request.form["Square_Footage"]),
        float(request.form["Num_Bedrooms"]),
        float(request.form["Num_Bathrooms"]),
        float(request.form["Year_Built"]),
        float(request.form["Lot_Size"]),
        float(request.form["Garage_Size"]),
        float(request.form["Neighborhood_Quality"]),
    ]

    # Make prediction
    final_features = np.array([features])
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Estimated Property Value: ${output:,.2f}",
    )
  except Exception as e:
    return render_template(
        "index.html", prediction_text=f"Error in processing: {str(e)}"
    )


if __name__ == "__main__":
  app.run(debug=True)
