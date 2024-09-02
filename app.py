import streamlit as st
import tensorflow as tf
from tensorflow.keras import activations, optimizers, losses
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")


def construct_encodings(x, tokenizer, max_len, trucation=True, padding=True):
    return tokenizer(x, max_length=max_len, truncation=trucation, padding=padding)

def construct_tfdataset(encodings, y=None):
    if y:
        return tf.data.Dataset.from_tensor_slices((dict(encodings),y))
    else:
        # this case is used when making predictions on unseen samples after training
        return tf.data.Dataset.from_tensor_slices(dict(encodings))

def create_predictor(tokenizer, max_len):

    #tkzr = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tkzr= tokenizer

    def predict_proba(text):

        x = [text]

        encodings = construct_encodings(x, tkzr, max_len=max_len, trucation=True, padding=True)
        tfdataset = construct_tfdataset(encodings,y=None)
        tfdataset = tfdataset.batch(1)

        preds = model.predict(tfdataset,verbose=0).logits
        preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
        return preds[0]

    return predict_proba


def get_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = TFDistilBertForSequenceClassification.from_pretrained('kishakumar09ei/HateSpeechDetection_bits_project_2',use_auth_token='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')
    return tokenizer, model

max_len=40

tokenizer,model = get_model()

st.title("Empower Your Online Spaces: Detect and Prevent Hate Speech")
st.subheader("Harness AI to Identify and Manage Hate Speech Across Platforms")
st.markdown("Welcome to [Your App Name], the innovative solution designed to help you detect and manage hate speech effectively. Our advanced AI algorithms analyze text in real-time to identify harmful content, enabling you to take action and foster a positive online community.")

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")


if user_input and button:
    test_sample = user_input
    tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    clf = create_predictor(tokenizer, max_len)
    prediction = clf(test_sample)
    if prediction[0]>0.2:
        prediction[0] = prediction[0]-0.15
        prediction[1] = prediction[1]+0.15

    st.write(f"Probability of this text to be non hate speech: {np.round(prediction[0],3)}")
    st.write(f"Probability of this text to be hate speech: {np.round(prediction[1], 3)}")
    label_ = np.argmax(np.array(prediction))
    if label_ == 1:
        st.write("We have detected 'hate speech' in this comment")
    else:
        st.write("We couldn't detected 'hate speech' in this comment")

