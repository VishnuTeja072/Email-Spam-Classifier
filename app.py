import streamlit as st
from src.predict import predict_email

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Classifier")
st.write("Classify emails as **Spam** or **Not Spam** using Machine Learning.")

email_text = st.text_area("✉️ Paste Email Content", height=200)

if st.button("🔍 Classify"):
    if email_text.strip() == "":
        st.warning("Please enter an email message.")
    else:
        try:
            label, confidence = predict_email(email_text)
        except Exception as e:
            st.error(f"Error classifying message: {e}")
        else:
            if label == 1:
                st.error(f"🚨 SPAM Email ({confidence}% confidence)")
            else:
                st.success(f"✅ NOT SPAM ({confidence}% confidence)")
