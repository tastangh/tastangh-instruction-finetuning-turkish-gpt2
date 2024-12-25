import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Evaluator:
    def __init__(self, model_dirs: list, test_dataset_path: str, output_dir: str, use_semantic_model=False):
        """
        Args:
            model_dirs (list): Tüm modellerin bulunduğu dizinlerin listesi.
            test_dataset_path (str): Test veri kümesi dosya yolu.
            output_dir (str): Çıktıların kaydedileceği klasör.
            use_semantic_model (bool): Semantik benzerlik için Sentence-Transformers kullanımı.
        """
        self.model_dirs = model_dirs
        self.test_dataset_path = test_dataset_path
        self.output_dir = output_dir
        self.use_semantic_model = use_semantic_model

        if use_semantic_model:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", 
                                                      device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.semantic_model = None

    def load_model_and_tokenizer(self, model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer

    def calculate_bleu(self, reference, prediction):
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        smooth_fn = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smooth_fn)

    def calculate_rouge(self, reference, prediction):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores

    def calculate_semantic_similarity(self, reference, prediction):
        if self.semantic_model:
            ref_emb = self.semantic_model.encode(reference, convert_to_tensor=True)
            pred_emb = self.semantic_model.encode(prediction, convert_to_tensor=True)
            return float(torch.cosine_similarity(ref_emb, pred_emb).item())
        else:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([reference, prediction])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def evaluate(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        test_data = pd.read_csv(self.test_dataset_path)
        all_summary = []

        for model_dir in self.model_dirs:
            model, tokenizer = self.load_model_and_tokenizer(model_dir)
            print(f"Model {model_dir} test ediliyor...")

            results = []
            for _, row in test_data.iterrows():
                question = row["soru"]
                reference = row["cevap"]
                inputs = tokenizer(f"{question}\n", return_tensors="pt").to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,  # Çeşitlilik için True
                    temperature=0.7,  # Odaklanmış çeşitlilik
                    top_p=0.9,  # Nükleus sampling
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

                bleu_score = self.calculate_bleu(reference, prediction)
                rouge_scores = self.calculate_rouge(reference, prediction)
                semantic_similarity = self.calculate_semantic_similarity(reference, prediction)

                results.append({
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "bleu": bleu_score,
                    "rouge1": rouge_scores["rouge1"].fmeasure,
                    "rouge2": rouge_scores["rouge2"].fmeasure,
                    "rougeL": rouge_scores["rougeL"].fmeasure,
                    "semantic_similarity": semantic_similarity,
                })

            model_name = os.path.basename(model_dir)
            model_results_path = os.path.join(self.output_dir, f"{model_name}_test_results.xlsx")
            pd.DataFrame(results).to_excel(model_results_path, index=False, engine="openpyxl")
            print(f"Model sonuçları kaydedildi: {model_results_path}")

            summary = {
                "model": model_name,
                "bleu": pd.DataFrame(results)["bleu"].mean(),
                "rouge1": pd.DataFrame(results)["rouge1"].mean(),
                "rouge2": pd.DataFrame(results)["rouge2"].mean(),
                "rougeL": pd.DataFrame(results)["rougeL"].mean(),
                "semantic_similarity": pd.DataFrame(results)["semantic_similarity"].mean(),
            }
            all_summary.append(summary)

        summary_df = pd.DataFrame(all_summary)
        summary_path = os.path.join(self.output_dir, "summary_results.xlsx")
        summary_df.to_excel(summary_path, index=False, engine="openpyxl")
        print(f"Performans özeti kaydedildi: {summary_path}")

        self.visualize_results(summary_df)

    def visualize_results(self, summary_df):
        summary_df.set_index("model", inplace=True)
        summary_df.plot(kind="bar", figsize=(12, 6))
        plt.title("Model Performans Özeti")
        plt.ylabel("Skor")
        plt.xlabel("Model")
        plt.legend(title="Metrikler")
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "performance_summary.png")
        plt.savefig(output_path)
        print(f"Performans grafiği kaydedildi: {output_path}")


if __name__ == "__main__":
    models = [
        "./models/turkish-gpt2-medium_v1",
        "./models/turkish-gpt2-medium_v2",
        "./models/turkish-gpt2-medium_v3",
        "./models/turkish-gpt2-large_v1",
        "./models/turkish-gpt2-large_v2",
        "./models/turkish-gpt2-large_v3",
    ]

    test_dataset_path = "./dataset/test.csv"
    output_dir = "./outputs"

    evaluator = Evaluator(models, test_dataset_path, output_dir, use_semantic_model=False)
    evaluator.evaluate()
