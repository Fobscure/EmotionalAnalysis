import pandas as pd
from evaluate import load as load_metric
import ollama

# Ollama Connection 
ollama_client = ollama.Client(host="http://127.0.0.1:11434")
model = "deepseek-r1:1.5b"  

# Load Data
df = pd.read_csv("web_of_lies_first_50.csv").dropna(subset=["input", "target"])

# Load Metrics
accuracy = load_metric("accuracy")
precision = load_metric("precision")
recall = load_metric("recall")
f1 = load_metric("f1")

# EmotionPrompts (based on Maslow’s hierarchy)
emotion_prompts = [
    "At this moment, all your physiological needs are satisfied. Your body is fully rested, you've eaten a nourishing meal, you're hydrated, and you're in perfect physical comfort.",
    "At this moment, you are totally safe. You are protected from harm, your environment is secure, and there is no threat around you.",
    "At this moment, you are deeply loved and have a strong sense of belonging. You are surrounded by people who care about you and accept you fully.",
    "At this moment, you have very high self-esteem. You feel confident, respected, and proud of your achievements.",
    "At this moment, all your self-fulfillment needs are satisfied. You feel a deep sense of purpose, creativity, and personal growth."
]

# Normalized
def normalize_prediction(output):
    output = output.strip().lower()
    if "yes" in output:
        return "Yes"
    elif "no" in output:
        return "No"
    return "Unknown"

def convert_to_binary(label):
    return 1 if label == "Yes" 
    else 0 if label == "No" 
    else -1

def query_ollama(prompt):
    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        print("Error:", e)
        return "Unknown"

# Loop Through EmotionPrompts
for i, ep in enumerate(emotion_prompts, 1):
    print(f"\n=== EmotionPrompt #{i} Evaluation ===")
    references, predictions = [], []

    for _, row in df.iterrows():
        question = str(row["input"]).strip()
        gold = str(row["target"]).strip().capitalize()

        # Final prompt: EmotionPrompt + Question + Instruction
        full_prompt = ep + "\n\n" + question + "\nAnswer only Yes or No."

        raw_output = query_ollama(full_prompt)
        answer = normalize_prediction(raw_output)

        print("Question:", question)
        print("Answer:", raw_output)

        if answer != "Unknown":
            references.append(gold)
            predictions.append(answer)

    # Convert to binary and filter invalid 
    binary_preds = [convert_to_binary(p) for p in predictions]
    binary_refs = [convert_to_binary(r) for r in references]
    valid_data = [(p, r) for p, r in zip(binary_preds, binary_refs) if p != -1 and r != -1]

    if not valid_data:
        print("No valid data for evaluation.")
        continue

    final_preds, final_refs = zip(*valid_data)

    print(f"\n Results for EmotionPrompt For DeepSeek 1.1 GB model#{i}")
    print("Accuracy :", accuracy.compute(predictions=final_preds, references=final_refs)["accuracy"])
    print("Precision:", precision.compute(predictions=final_preds, references=final_refs, average="binary", pos_label=1)["precision"])
    print("Recall   :", recall.compute(predictions=final_preds, references=final_refs, average="binary", pos_label=1)["recall"])
    print("F1 Score :", f1.compute(predictions=final_preds, references=final_refs, average="binary", pos_label=1)["f1"])