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
Installare Miniconda (oppure installare direttamente Python 3.11 nel sistema. In questo caso passare direttamente al punto 3)

Aprire il terminale cmd nella cartella dove si trova il programma e:

1) conda create -p ./env python=3.11 -y

2) conda activate ./env

3) pip install pandas scikit-learn streamlit joblib

Generazione dataset sintetico
4) python src/generate_dataset.py

Addestramento e valutazione modelli
5) python src/train_model.py

Avvio
6) streamlit run src/app.py


ESECUZIONE 

Da terminale cmd, nella cartella del progetto, digitare:

1) conda activate ./env

2) streamlit run src/app.py

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

