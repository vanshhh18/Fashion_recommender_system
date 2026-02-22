
#  Fashion Recommender System (Deep Learning Based)

A Deep Learning based Fashion Recommender System that suggests visually similar clothing items using feature embeddings extracted from a pre-trained ResNet model.

---

##  Overview

This project implements a Content-Based Recommendation System using Convolutional Neural Networks (CNN).

The system:
- Extracts deep image features using ResNet (pre-trained on ImageNet)
- Converts images into numerical embeddings
- Computes Euclidean distance between embeddings
- Recommends visually similar fashion products

---

##  Model Architecture

- Pre-trained ResNet (without top classification layer)
- Global Max Pooling layer
- Feature vector normalization
- Euclidean Distance for similarity comparison

---

##  Features

- Image-based fashion recommendations
- Deep feature extraction using TensorFlow + ResNet
- Fast similarity computation
- Clean Web Interface
- Scalable embedding storage

---

##  Tech Stack

- Python
- TensorFlow
- Keras
- ResNet50
- NumPy
- Scikit-learn
- Streamlit / Flask
- HTML, CSS

---

##  Project Structure

fashion_ml

images                   - Dataset images


embeddings.pkl           - Stored feature vectors


filenames.pkl            - Image filenames


app.py                   - Main application


main.py                  - Streamlit operation


test.py                  - Testing operation


uploads                  - Testing images


README.md

---

##  How It Works

1. Load images from dataset  
2. Preprocess images (resize, normalize)  
3. Extract features using ResNet (without final classification layer)  
4. Store embeddings  
5. Compute Euclidean distance between selected item and dataset  
6. Return Top-K similar items  

---

##  Similarity Metric

Euclidean Distance:

distance = √ Σ (x1 - x2)²  

Smaller distance → Higher similarity  

---

##  Learning Outcomes

- Practical use of Transfer Learning
- Feature extraction using CNN
- Understanding embedding spaces
- Building end-to-end ML applications
- Model deployment basics
