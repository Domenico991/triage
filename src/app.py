import streamlit as st
import joblib
import pandas as pd
import re

# Load models
vectorizer = joblib.load("models/vectorizer.pkl")
cat_model = joblib.load("models/category_model.pkl")
pri_model = joblib.load("models/priority_model.pkl")

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zàèéìòù ]", " ", text)
    return text

st.title("Triage dei Ticket")

# --- FORM ---
with st.form("ticket_form"):
    title = st.text_input("Titolo del ticket")
    body = st.text_area("Descrizione del ticket")
    submit = st.form_submit_button("Classifica ticket")

if submit:
    if title.strip() == "" and body.strip() == "":
        st.error("Inserire almeno titolo o descrizione.")
    else:
        text = clean(title + " " + body)
        X = vectorizer.transform([text])

        cat = cat_model.predict(X)[0]
        pri = pri_model.predict(X)[0]

        st.success(f"Categoria prevista: {cat}")
        st.warning(f"Priorità suggerita: {pri}")

        # Parole più influenti
        feature_names = vectorizer.get_feature_names_out()
        class_index = list(cat_model.classes_).index(cat)
        coefs = cat_model.coef_[class_index]

        top_idx = coefs.argsort()[-5:][::-1]
        keywords = [feature_names[i] for i in top_idx]

        st.subheader("Parole più influenti")
        st.write(keywords)

# --- BATCH ---
st.divider()
st.subheader("Predizione batch da CSV")

file = st.file_uploader("Carica un CSV con colonne: title, body", type="csv")

if file:
    df = pd.read_csv(file)
    texts = (df["title"] + " " + df["body"]).apply(clean)

    Xb = vectorizer.transform(texts)
    df["pred_category"] = cat_model.predict(Xb)
    df["pred_priority"] = pri_model.predict(Xb)

    st.dataframe(df.head())

    df.to_csv("data/predictions.csv", index=False)
    st.success("Predizioni salvate in data/predictions.csv")
