import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load data
df = pd.read_csv("data/tickets_synthetic.csv")

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zàèéìòù ]", " ", text)
    return text

X = (df["title"] + " " + df["body"]).apply(clean)

y_cat = df["category"]
y_pri = df["priority"]

X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
    X, y_cat, y_pri, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

cat_model = LogisticRegression(max_iter=200)
cat_model.fit(X_train_vec, y_cat_train)

pri_model = LogisticRegression(max_iter=200)
pri_model.fit(X_train_vec, y_pri_train)

cat_pred = cat_model.predict(X_test_vec)
pri_pred = pri_model.predict(X_test_vec)

print("Categoria – Accuracy:", accuracy_score(y_cat_test, cat_pred))
print("Categoria – F1 macro:", f1_score(y_cat_test, cat_pred, average="macro"))
print(confusion_matrix(y_cat_test, cat_pred))

print("Priorità – Accuracy:", accuracy_score(y_pri_test, pri_pred))
print("Priorità – F1 macro:", f1_score(y_pri_test, pri_pred, average="macro"))

joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(cat_model, "models/category_model.pkl")
joblib.dump(pri_model, "models/priority_model.pkl")