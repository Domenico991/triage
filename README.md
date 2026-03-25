# Triage automatico dei ticket con Machine Learning

---

## Requisiti

- Python >= 3.9
- Librerie:
  - pandas
  - scikit-learn
  - streamlit
  - joblib

Installazione rapida:

```bash
pip install pandas scikit-learn streamlit joblib
```

---

## Struttura del progetto

```
project/
│
├── data/
│   ├── tickets_synthetic.csv      # dataset generato
│   └── predictions.csv            # output batch
│
├── src/
│   ├── generate_dataset.py        # generatore dataset
│   ├── train_model.py             # training e valutazione ML
│   └── app.py                     # dashboard Streamlit
│
├── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   └── vectorizer.pkl
│
├── README.md
└── report.md
```

---

## Esecuzione del prototipo

### 1. Generazione dataset sintetico

```bash
python src/generate_dataset.py
```

Output:
- `data/tickets_synthetic.csv`

---

### 2. Addestramento e valutazione modelli

```bash
python src/train_model.py
```

Output:
- metriche stampate a console (accuracy, F1, confusion matrix)
- modelli salvati in `models/`

---

### 3. Avvio dashboard

```bash
streamlit run src/app.py
```

Funzionalità:
- inserimento manuale di un ticket;
- visualizzazione categoria e priorità previste;
- parole più influenti per la decisione;
- caricamento CSV per predizioni batch con esportazione.

---

## Output principali

- Dataset sintetico condivisibile
- Modelli ML serializzati
- CSV di predizioni batch
- Dashboard interattiva

---

## Note

Il progetto è pensato come **proof-of-concept** didattico, facilmente estendibile a dati reali previo opportuno preprocessing e validazione.

