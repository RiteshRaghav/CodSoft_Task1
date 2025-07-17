# Sales Prediction using Machine Learning

This project demonstrates how to predict sales using machine learning techniques in Python. It covers data loading, exploratory data analysis (EDA), model building, and evaluation.

## Project Structure
- `data/advertising.csv`: Advertising dataset used for sales prediction
- `sales_prediction.ipynb`: Jupyter notebook with step-by-step workflow
- `sales_prediction.py`: Main Python script for data analysis and modeling
- `requirements.txt`: Project dependencies
- `best_random_forest_model.pkl`: Saved trained model

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook sales_prediction.ipynb
   ```
3. Follow the steps in the notebook.

## Dataset
The dataset (`data/advertising.csv`) contains the following columns:
- **TV**: Advertising budget spent on TV (in thousands of dollars)
- **Radio**: Advertising budget spent on Radio (in thousands of dollars)
- **Newspaper**: Advertising budget spent on Newspaper (in thousands of dollars)
- **Sales**: Sales of the product (in thousands of units)

Make sure your CSV file matches this structure for the code to work correctly.
