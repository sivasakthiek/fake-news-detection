# Fake News Detector

A simple Fake News Detector web application built with Streamlit and Python, which classifies news articles as Real or Fake using a pre-trained machine learning model.

## Features

User-friendly interface using Streamlit

Input a news article and check if it is Real or Fake

Uses a pre-trained vectorizer and logistic regression model

Displays results with clear success/error messages

## Prerequisites

Python 3.10+

Install required packages:

pip install streamlit scikit-learn joblib


## Project Files

app.py → The main Streamlit app

vectorizer.jb → Saved TF-IDF/CountVectorizer

lr_model.jb → Saved Logistic Regression model

Ensure vectorizer.jb and lr_model.jb are in the same folder as app.py.

## How to Run

Open a terminal in the project folder

Run the Streamlit app:

streamlit run app.py


Your browser will open the app

Enter a news article in the text area

Click Check News to see if it is Real or Fake

## Example Inputs

Real news example:

NASA announced that the James Webb Telescope has captured images of a distant galaxy 13 billion light-years away.


Fake news example:

Eating chocolate cake daily will make you live forever.


## Notes

The accuracy of detection depends on the training dataset used for the model

Train the model on relevant data for better results
