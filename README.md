# Email Spam Detector

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A simple machine learning project to classify emails as "Spam" or "Ham" using Python and Scikit-learn.

This project demonstrates a complete machine learning workflow for a text classification problem. It preprocesses email data, trains a Logistic Regression model using a TF-IDF vectorizer, and provides a mechanism to classify new emails.

## ✨ Key Features

*   **Data Preprocessing:** Cleans and prepares raw text data for modeling.
*   **TF-IDF Vectorization:** Converts text messages into meaningful numerical feature vectors.
*   **Logistic Regression Model:** A simple yet effective classification algorithm for this task.
*   **Model Persistence:** Saves the trained model and vectorizer to disk, so you don't have to retrain every time.
*   **Modular Code:** The script is structured with functions for clarity and reusability.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8 or later
*   pip (Python package installer)
*   The dataset `mail_data.csv` in the project's root directory.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

The script can be run directly from the command line.

**First Run (Training the Model):**

The first time you run the script, it will train the model using `mail_data.csv` and save the `spam_detector_model.joblib` and `tfidf_vectorizer.joblib` files.

```bash
python code.py
```

You will see output like this:
```
Model not found. Training a new model...
Accuracy on training data: 0.9966
Accuracy on test data: 0.9659
Model saved to spam_detector_model.joblib
Vectorizer saved to tfidf_vectorizer.joblib

--- Email Classification ---
Email: 'Dear Lucky Winner, We are Pleased to inform you that your was randoml...'
Prediction: Spam mail
```

**Subsequent Runs (Using the Saved Model):**

On subsequent runs, the script will load the saved model and vectorizer to perform predictions instantly. To classify a different email, you can modify the `input_your_mail` variable inside the `main()` function in `code.py`.

## 📜 License

This project is licensed under the MIT License.

