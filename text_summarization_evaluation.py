!pip install transformers datasets torch rouge_score nltk pandas tqdm
import os
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2Tokenizer, BartTokenizer, BartForConditionalGeneration
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import pandas as pd
from tqdm import tqdm
!pip install openai==0.28
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import openai
import torch
print(torch.cuda.is_available())
from google.colab import userdata

# Download required NLTK resources
nltk.download('punkt', quiet=True)
!huggingface-cli login

# Load cnn_dailymail dataset
dataset = load_dataset("abisee/cnn_dailymail","3.0.0")
df = dataset['test'].to_pandas()

# Select a subset of the data for evaluation (adjust as needed)
num_samples = 100
input_texts = df['article'][:num_samples].tolist()
reference_texts = df['highlights'][:num_samples].tolist()

# Function to evaluate gpt-2 model
def evaluate_gpt2_model(input_texts, reference_texts, model_name="openai-community/gpt2-large"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the pad token to the eos token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    if torch.cuda.is_available():
        model = model.to('cuda')

    results = []

    for input_text, reference_text in tqdm(zip(input_texts, reference_texts), total=len(input_texts), desc=f"Evaluating {model_name}"):
        try:
            prompt = f"Summarize the following text in 30 to 40 words:\n\n{input_text}\n\nSummary:"
            inputs = tokenizer.encode_plus(
                prompt,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding='max_length'
            )

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            summary_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=120,
                min_length=80,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            if "Summary:" in generated_output:
                generated_summary = generated_output.split("Summary:")[1].strip()
            else:
                generated_summary = generated_output.strip()

            if not generated_summary:
                generated_summary = "No summary generated."

            results.append(calculate_scores(generated_summary, reference_text, model_name))
        except Exception as e:
            print(f"Error processing input for {model_name}: {e}")
            results.append({
                'model': model_name,
                'reference_text': reference_text,
                'generated_output': "Error occurred",
                'rouge1': 0,
                'rouge2': 0,
                'rougeL': 0,
                'bleu': 0
            })

    return results

# Define BART model
def evaluate_bart_model(input_texts, reference_texts, model_name="facebook/bart-large-cnn"):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.to('cuda')

    results = []

    for input_text, reference_text in tqdm(zip(input_texts, reference_texts), total=len(input_texts), desc=f"Evaluating {model_name}"):
        try:
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')

            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=120,
                min_length=80,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            generated_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            results.append(calculate_scores(generated_output, reference_text, model_name))
        except Exception as e:
            print(f"Error processing input for {model_name}: {e}")

    return results

# Initialize ROUGE scorer and BLEU smoothing function
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu_smoothing = SmoothingFunction().method4

def evaluate_t5_model(model_name, input_texts, reference_texts):
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=model_name, device=device)
    results = []

    for input_text, reference_text in tqdm(zip(input_texts, reference_texts), total=len(input_texts), desc=f"Evaluating {model_name}"):
        try:
            generated_output = summarizer(
                input_text,
                max_length=120,
                min_length=80,
                do_sample=False
            )[0]['summary_text']

            results.append(calculate_scores(generated_output, reference_text, model_name))
        except Exception as e:
            print(f"Error processing input for {model_name}: {e}")

    return results

def calculate_scores(generated_output, reference_text, model_name):
    rouge_scores = scorer.score(reference_text, generated_output)

    reference_tokens = nltk.word_tokenize(reference_text)
    generated_tokens = nltk.word_tokenize(generated_output)
    bleu = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=bleu_smoothing)

    return {
        'model': model_name,
        'reference_text': reference_text,
        'generated_output': generated_output,
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu
    }

# Evaluate GPT-2
print("\nEvaluating gpt2-large...")
gpt2_results = evaluate_gpt2_model(input_texts, reference_texts, "openai-community/gpt2-large")

# Evaluate BART model
print("\nEvaluating BART-large-CNN...")
bart_results = evaluate_bart_model(input_texts, reference_texts, "facebook/bart-large-cnn")

# Evaluate T5-large
print("\nEvaluating T5-large...")
t5_results = evaluate_t5_model("google-t5/t5-large", input_texts, reference_texts)

# Combine all results
all_results = t5_results + bart_results + gpt2_results

# Convert results to DataFrame for easy analysis
results_df = pd.DataFrame(all_results)

# Calculate average scores for each model
avg_scores = results_df.groupby('model').agg({
    'rouge1': 'mean',
    'rouge2': 'mean',
    'rougeL': 'mean',
    'bleu': 'mean'
}).round(4)

print("\nAverage Scores:")
print(avg_scores)

# Save detailed results to CSV
results_df.to_csv("summarization_evaluation_results.csv", index=False)
print("\nDetailed results saved to 'summarization_evaluation_results.csv'")

