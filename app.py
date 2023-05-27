import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
     

# Load the pre-trained model and tokenizer
model_path = 'Toxic_detect_Model'  # Update with the actual path to your saved model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer =BertTokenizer.from_pretrained('bert-base-uncased')

# Function to process the input text and generate predictions
def predict(text):
    inputs = tokenizer(text,padding = True, truncation = True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()
    predicted_label_index = np.argmax(predictions)
    return predicted_label_index

# Streamlit application
def main():
    st.title(" Predicting Short Texts with a BERT Model Fine-Tuned on a Hate Speech Dataset from a White Supremacy Forum")

    # Input text box
    input_text = st.text_area("Enter text", height=50)

    # Process text and generate predictions on button click
    if st.button("Predict"):
        if input_text:
            predicted_labels= predict(input_text)
            if predicted_labels == 0:
                st.write("The text is Non- Toxic")
            else:
                st.write("The Text is Toxic")
            
        else:
            st.write("Please enter some text.")

if __name__ == '__main__':
    main()
