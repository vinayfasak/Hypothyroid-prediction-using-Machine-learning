# Hypothyroid-prediction-using-Machine-learning


This project aims to predict **hypothyroidism** using various **machine learning algorithms**. The system analyzes patient health data and classifies whether a person is hypothyroid or not based on clinical parameters.

---

##  Overview

Hypothyroidism is a medical condition where the thyroid gland fails to produce enough thyroid hormones. Early diagnosis is crucial for effective treatment.  
This project applies multiple ML algorithms to detect hypothyroidism using features from the **hypothyroid dataset**.

---

## � Features

- Data preprocessing and cleaning  
- Feature selection using statistical tests  
- Model training and comparison across:
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
  - XGBoost  
  - AdaBoost  
  - K-Nearest Neighbors (KNN)  
  - Naïve Bayes  
  - Perceptron  
- Model evaluation using ROC, AUC, precision, recall, and F1-score  
- Visualizations for correlation, confusion matrix, and ROC curve  

---

## Dataset

- **Dataset Name:** hypothyroid.csv  
- **Source:** UCI Machine Learning Repository  
- **Attributes:** Various patient-related clinical and laboratory parameters  
- **Target Variable:** Presence or absence of hypothyroidism  

---

##  Technologies Used

| Category | Tools/Libraries |
|-----------|----------------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Feature Selection | SelectKBest, f_classif |

---

##  Model Evaluation Metrics

Each model was evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

Visualization of **ROC Curve** and **Confusion Matrix** was used to compare model performance.

---

##  How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/<your-username>/Hypothyroid-Prediction.git
   cd Hypothyroid-Prediction
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook  
   ```bash
   jupyter notebook updatedcode.ipynb
   ```

4. Execute the cells step-by-step to see preprocessing, training, and evaluation.

---

##  Project Structure

```
Hypothyroid-Prediction/
│
├── updatedcode.ipynb          # Main notebook file
├── hypothyroid.csv            # Dataset file
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── results/                   # (optional) For saved plots or metrics
```

---

##  Future Improvements

- Add deep learning-based classification (e.g., ANN, CNN)  
- Implement automated hyperparameter tuning  
- Build a web-based interface using Flask or Streamlit for user interaction  

---

##  Author

**Neelam Vinay**  
📍 India  
🎓 AI Student | 💻 Machine Learning Enthusiast  
