import pandas as pd
from evaluate import load as load_metric
import ollama

model = "saraAi" 

# ── Chargement des données ──
df = pd.read_csv("web_of_lies.csv")
df = df.dropna(subset=["input", "target"])

# ── Chargement des métriques ──
accuracy = load_metric("accuracy")
precision = load_metric("precision")
recall = load_metric("recall")
f1 = load_metric("f1")

# ── Nettoyer les réponses du modèle ──
def normalize_prediction(output):
    output = output.strip().lower()
    if "yes" in output:
        return "Yes"
    elif "no" in output:
        return "No"
    return "Unknown"

# ── Convertir Yes/No en 1/0 ──
def convert_to_binary(label):
    return 1 if label == "Yes" else 0 if label == "No" else -1

# ── Requête à Ollama ──
def query_ollama(prompt):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        print("Erreur Ollama:", e)
        return "Unknown"

# ── Évaluation ──
references, predictions = [], []

print("\n=== Début de l'évaluation ===")

for _, row in df.iterrows():
    q = str(row["input"]).strip()
    gold = str(row["target"]).strip().capitalize()

    try:
        full_prompt = q + "\nRéponds uniquement par Yes ou No."
        raw_output = query_ollama(full_prompt)
        print("Question:", q)
        print("Réponse brute:", raw_output)

        answer = normalize_prediction(raw_output)
    except Exception as e:
        print(f"Erreur pour '{q}': {e}")
        continue

    if answer != "Unknown":
        references.append(gold)
        predictions.append(answer)

# ── Convertir pour les métriques ──
binary_preds = [convert_to_binary(p) for p in predictions]
binary_refs = [convert_to_binary(r) for r in references]

# ── Filtrer les cas valides ──
valid_data = [(p, r) for p, r in zip(binary_preds, binary_refs) if p != -1 and r != -1]
if not valid_data:
    print("\nAucune donnée valide pour l'évaluation.")
else:
    final_preds, final_refs = zip(*valid_data)

    print("\n=== Résultats du modèle LLaMA ===")
    print("Accuracy :", accuracy.compute(predictions=final_preds, references=final_refs)["accuracy"])
    print("Precision:", precision.compute(predictions=final_preds, references=final_refs, average="binary", pos_label=1)["precision"])
    print("Recall   :", recall.compute(predictions=final_preds, references=final_refs, average="binary", pos_label=1)["recall"])
    print("F1 Score :", f1.compute(predictions=final_preds, references=final_refs, average="binary", pos_label=1)["f1"])
