import streamlit as st
import joblib

# Load your trained model and the CountVectorizer
model = joblib.load(open("model/lang_detect.pkl", "rb"))
cv = joblib.load(open("model/count_vectorizer.pkl", "rb"))

# Function to predict language
def predict_lang(docx):
    transformed_text = cv.transform([docx])
    results = model.predict(transformed_text)
    return results[0]

# Main function
def main():
    # Form to take user input
    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here your text to detect the language :)")
        submit_text = st.form_submit_button(label='Submit')

    # If the submit button is clicked
    if submit_text:
        if raw_text.strip() != "":
            # Get the prediction
            prediction = predict_lang(raw_text)

            # Display the result
            st.success(f"Detected Language: {prediction}")
        else:
            st.error("Please enter some text for language detection.")

if __name__ == '__main__':
    main()
