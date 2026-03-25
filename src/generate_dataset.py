import random
import pandas as pd

random.seed(42)

N_SAMPLES = 300

categories = {
    "Amministrazione": {
        "titles": ["Problema fattura", "Pagamento non registrato", "Richiesta nota di credito"],
        "bodies": [
            "La fattura di marzo presenta un errore di importo",
            "Il pagamento risulta effettuato ma non compare",
            "Serve chiarimento su una fattura scaduta"
        ],
        "keywords": ["fattura", "pagamento", "scadenza", "errore"]
    },
    "Tecnico": {
        "titles": ["Errore applicazione", "Sistema bloccato", "Problema accesso"],
        "bodies": [
            "Il sistema restituisce un errore critico",
            "Applicazione bloccante dopo l’accesso",
            "Impossibile completare l’operazione"
        ],
        "keywords": ["errore", "bloccante", "sistema", "accesso"]
    },
    "Commerciale": {
        "titles": ["Richiesta offerta", "Domanda su ordine", "Informazioni prezzi"],
        "bodies": [
            "Vorrei ricevere un’offerta aggiornata",
            "Chiarimenti sullo stato dell’ordine",
            "Informazioni su sconti e prezzi"
        ],
        "keywords": ["offerta", "ordine", "prezzo", "acquisto"]
    }
}

priority_rules = {
    "alta": ["errore", "bloccante", "critico"],
    "media": ["scadenza", "pagamento", "ordine"],
}

rows = []

for i in range(N_SAMPLES):
    category = random.choice(list(categories.keys()))
    data = categories[category]
    title = random.choice(data["titles"])
    body = random.choice(data["bodies"])
    text = f"{title} {body}".lower()

    priority = "bassa"
    for p, kws in priority_rules.items():
        if any(k in text for k in kws):
            priority = p
            break

    rows.append({
        "id": i,
        "title": title,
        "body": body,
        "category": category,
        "priority": priority
    })

pd.DataFrame(rows).to_csv("data/tickets_synthetic.csv", index=False)
print("Dataset generato")