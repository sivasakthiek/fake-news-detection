import streamlit as st
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# =======================
# Function to create a dummy model if files are missing
# =======================
def create_dummy_model():
    sample_texts = [
        "library open late", "free laptops", "exams cancelled", "scholarship available"
    ]
    labels = [1, 0, 0, 1]  # 1 = Real, 0 = Fake

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(sample_texts)
    model = LogisticRegression()
    model.fit(X_vec, labels)

    # Save the dummy model for future runs
    joblib.dump(vectorizer, "vectorizer.jb")
    joblib.dump(model, "lr_model.jb")

    return vectorizer, model

# =======================
# Load model and vectorizer
# =======================
if os.path.exists("vectorizer.jb") and os.path.exists("lr_model.jb"):
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
else:
    vectorizer, model = create_dummy_model()
    st.warning("⚠️ Model files not found. Using a dummy model for testing.")

# =======================
# Streamlit App UI
# =======================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write(
    "Enter a news article below to check whether it is **Real** or **Fake**."
)

input_text = st.text_area("News Article:", "", height=150)

st.markdown("**Sample news to test:**")
st.markdown("""
- The university library will stay open until midnight for exam preparation.  
- Free laptops will be distributed to students attending the campus fest.  
- Exams are cancelled this semester due to heavy rain.  
- Scholarship applications are open until the end of the month.  
""")

if st.button("Check News"):
    if input_text.strip():
        transformed_input = vectorizer.transform([input_text])
        prediction = model.predict(transformed_input)

        if prediction[0] == 1:
            st.success("✅ The News is Real!")
        else:
            st.error("❌ The News is Fake!")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
