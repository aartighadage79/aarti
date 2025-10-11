{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9951908-7884-4640-9a7f-96246e518456",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained model\n",
    "with open(\"road_accident_model.pkl\", \"rb\") as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "st.title(\"🚦 Road Accident Severity Predictor\")\n",
    "\n",
    "# Example input fields (update based on your model's actual features)\n",
    "feature1 = st.number_input(\"Enter Feature 1 (e.g., Speed)\", value=50.0)\n",
    "feature2 = st.number_input(\"Enter Feature 2 (e.g., Age)\", value=30.0)\n",
    "feature3 = st.number_input(\"Enter Feature 3\", value=1.0)\n",
    "feature4 = st.number_input(\"Enter Feature 4\", value=1.0)\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    input_data = np.array([[feature1, feature2, feature3, feature4]])\n",
    "    prediction = model.predict(input_data)\n",
    "    st.success(f\"🎯 Predicted Class: {int(prediction[0])}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "",
   "name": ""
  },
  "language_info": {
   "name": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
#asdf
