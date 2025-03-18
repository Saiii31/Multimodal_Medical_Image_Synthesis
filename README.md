# Multimodal Medical Image Synthesis for Disease Prediction and its Analysis using Explainable AI

## Overview
This project focuses on generating medical images for disease prediction using multimodal learning techniques. By integrating image and text data, the system enhances diagnostic accuracy and leverages Explainable AI (XAI) to improve interpretability and trust in medical models.

## Features
- **Multimodal Learning:** Combines medical images and textual descriptions to improve disease prediction.
- **Synthetic Image Generation:** Uses deep learning models to generate realistic medical images.
- **Explainable AI:** Enhances model transparency and interpretability to gain trust in medical predictions.
- **Disease Classification:** Predicts diseases with a high accuracy score.

## Tech Stack
- **Machine Learning** (TensorFlow, PyTorch)
- **Deep Learning** (CNN, GANs, Transformers)
- **Explainable AI** (SHAP, LIME)
- **Python** (NumPy, Pandas, Matplotlib, OpenCV)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/multimodal-medical-synthesis.git
   cd multimodal-medical-synthesis
   ```
2. Create a virtual environment:
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preparation:**
   - Place medical images in `data/images/`.
   - Store disease descriptions in `data/text/`.
   - Use `preprocess.py` to clean and structure the data.

2. **Training the Model:**
   ```sh
   python train.py --epochs 50 --batch_size 32
   ```

3. **Generating Synthetic Medical Images:**
   ```sh
   python generate_images.py --input sample_text.txt
   ```

4. **Explainability Analysis:**
   ```sh
   python explainability.py --image generated_image.png
   ```

## Results
- Achieved **SSIM score of 0.71** for synthetic image quality.
- Improved disease prediction accuracy through **multimodal data fusion**.
- Enhanced trust in AI-driven diagnosis using **explainability techniques**.

