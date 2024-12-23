import os  # Klasör kontrolü ve oluşturma için gerekli
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util


class Evaluator:
    def __init__(self, model_dirs: list, test_dataset_path: str, output_excel: str):
        """
        Args:
            model_dirs (list): Tüm modellerin bulunduğu dizinlerin listesi.
            test_dataset_path (str): Test veri kümesi dosya yolu.
            output_excel (str): Çıktıların kaydedileceği Excel dosya yolu.
        """
        self.model_dirs = model_dirs
        self.test_dataset_path = test_dataset_path
        self.output_excel = output_excel
        self.semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def load_model_and_tokenizer(self, model_dir):
        """
        Model ve tokenizer'ı verilen dizinden yükler.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer

    def calculate_bleu(self, reference, prediction):
        """
        BLEU skorunu hesaplar.
        """
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        smooth_fn = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smooth_fn)

    def calculate_rouge(self, reference, prediction):
        """
        ROUGE skorlarını hesaplar.
        """
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores

    def calculate_semantic_similarity(self, reference, prediction):
        """
        Semantik benzerliği hesaplar.
        """
        ref_emb = self.semantic_model.encode(reference, convert_to_tensor=True)
        pred_emb = self.semantic_model.encode(prediction, convert_to_tensor=True)
        return util.cos_sim(ref_emb, pred_emb).item()

    def evaluate(self):
        """
        Tüm modelleri test eder ve sonuçları metriklerle birlikte tabloya yazar.
        """
        # Klasör yoksa oluştur
        output_dir = os.path.dirname(self.output_excel)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Çıktı klasörü oluşturuldu: {output_dir}")

        # Test veri kümesini yükle
        test_data = pd.read_csv(self.test_dataset_path)
        results = []

        # Her modeli sırayla test et
        for idx, model_dir in enumerate(self.model_dirs, start=1):
            model, tokenizer = self.load_model_and_tokenizer(model_dir)
            print(f"Model {model_dir} test ediliyor...")

            for _, row in test_data.iterrows():
                question = row["soru"]
                reference = row["cevap"]
                inputs = tokenizer(f"{question}\n", return_tensors="pt").to(model.device)

                # Yanıt üretimi optimize edilmiş parametrelerle
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Metrik hesaplamaları
                bleu_score = self.calculate_bleu(reference, prediction)
                rouge_scores = self.calculate_rouge(reference, prediction)
                semantic_similarity = self.calculate_semantic_similarity(reference, prediction)

                # Sonuçları sakla
                results.append({
                    "model": model_dir,
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "bleu": bleu_score,
                    "rouge1": rouge_scores["rouge1"].fmeasure,
                    "rouge2": rouge_scores["rouge2"].fmeasure,
                    "rougeL": rouge_scores["rougeL"].fmeasure,
                    "semantic_similarity": semantic_similarity,
                })

            print(f"Model {model_dir} test işlemi tamamlandı.")

            # Sonuçları Excel dosyasına kaydet
            results_df = pd.DataFrame(results)
            results_df.to_excel(self.output_excel, index=False, engine="openpyxl")
            print(f"Sonuçlar {self.output_excel} dosyasına kaydedildi.")

            # Sonuçları TXT dosyasına kaydet
            txt_file_path = self.output_excel.replace(".xlsx", ".txt")  # Aynı isimli bir .txt dosyası oluştur
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                for result in results:
                    txt_file.write(
                        f"Model: {result['model']}\n"
                        f"Soru: {result['question']}\n"
                        f"Referans: {result['reference']}\n"
                        f"Tahmin: {result['prediction']}\n"
                        f"BLEU Skoru: {result['bleu']:.4f}\n"
                        f"ROUGE-1: {result['rouge1']:.4f}\n"
                        f"ROUGE-2: {result['rouge2']:.4f}\n"
                        f"ROUGE-L: {result['rougeL']:.4f}\n"
                        f"Semantik Benzerlik: {result['semantic_similarity']:.4f}\n"
                        f"{'-'*50}\n"
                    )
            print(f"Sonuçlar {txt_file_path} dosyasına kaydedildi.")


if __name__ == "__main__":
    # Test için kullanılacak modeller
    models = [
        "./models/turkish-gpt2-medium_v1",
        # "./models/turkish-gpt2-medium_v2",
        # "./models/turkish-gpt2-medium_v3",
        # "./models/turkish-gpt2-large_v1",
        # "./models/turkish-gpt2-large_v2",
        # "./models/turkish-gpt2-large_v3",
    ]

    # Test veri kümesi ve çıktı dosyası
    test_dataset_path = "./dataset/test.csv"
    output_excel = "./outputs/test_results.xlsx"

    # Değerlendirme işlemini başlat
    evaluator = Evaluator(models, test_dataset_path, output_excel)
    evaluator.evaluate()
