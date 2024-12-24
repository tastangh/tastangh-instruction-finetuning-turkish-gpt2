import os  # Klasör kontrolü ve oluşturma için gerekli
import pandas as pd
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

        # Semantik model veya alternatif benzerlik yöntemini seç
        if use_semantic_model:
            from sentence_transformers import SentenceTransformer, util
            self.semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.semantic_model = None

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
        Semantik benzerliği hesaplar (model tabanlı veya TF-IDF ile).
        """
        if self.semantic_model:
            # SentenceTransformer kullanımı
            ref_emb = self.semantic_model.encode(reference, convert_to_tensor=True)
            pred_emb = self.semantic_model.encode(prediction, convert_to_tensor=True)
            return float(torch.cosine_similarity(ref_emb, pred_emb).item())
        else:
            # TF-IDF ile Kosinüs Benzerliği
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([reference, prediction])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def evaluate(self):
        """
        Tüm modelleri test eder ve sonuçları metriklerle birlikte tabloya yazar.
        """
        # Klasör yoksa oluştur
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Çıktı klasörü oluşturuldu: {self.output_dir}")

        # Test veri kümesini yükle
        test_data = pd.read_csv(self.test_dataset_path)

        # Her modeli sırayla test et
        all_results = []
        for idx, model_dir in enumerate(self.model_dirs, start=1):
            model, tokenizer = self.load_model_and_tokenizer(model_dir)
            print(f"Model {model_dir} test ediliyor...")

            results = []
            for _, row in test_data.iterrows():
                question = row["soru"]
                reference = row["cevap"]
                inputs = tokenizer(f"{question}\n", return_tensors="pt").to(model.device)

                # Yanıt üretimi optimize edilmiş parametrelerle
                outputs = model.generate(
                    **inputs,
                    max_length=512,  # Tüm sekans uzunluğu
                    max_new_tokens=150,  # Yeni üretilen token uzunluğu
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
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "bleu": bleu_score,
                    "rouge1": rouge_scores["rouge1"].fmeasure,
                    "rouge2": rouge_scores["rouge2"].fmeasure,
                    "rougeL": rouge_scores["rougeL"].fmeasure,
                    "semantic_similarity": semantic_similarity,
                })

            # Model bazında sonuçları kaydet
            model_name = os.path.basename(model_dir)
            model_results_path = os.path.join(self.output_dir, f"{model_name}_results.xlsx")
            pd.DataFrame(results).to_excel(model_results_path, index=False, engine="openpyxl")
            print(f"Model sonuçları kaydedildi: {model_results_path}")

            # Genel sonuçlara ekle
            for result in results:
                result["model"] = model_name
            all_results.extend(results)

        # Genel performans özeti
        all_results_df = pd.DataFrame(all_results)
        summary_df = all_results_df.groupby("model").mean()
        summary_path = os.path.join(self.output_dir, "summary_results.xlsx")
        summary_df.to_excel(summary_path, index=True, engine="openpyxl")
        print(f"Performans özeti kaydedildi: {summary_path}")


if __name__ == "__main__":
    # Test için kullanılacak modeller
    models = [
        "./models/turkish-gpt2-medium_v1",
        "./models/turkish-gpt2-medium_v2",
        "./models/turkish-gpt2-medium_v3",
        "./models/turkish-gpt2-large_v1",
        "./models/turkish-gpt2-large_v2",
        "./models/turkish-gpt2-large_v3",
    ]

    # Test veri kümesi ve çıktı dosyası
    test_dataset_path = "./dataset/test.csv"
    output_dir = "./outputs"

    # Değerlendirme işlemini başlat
    evaluator = Evaluator(models, test_dataset_path, output_dir, use_semantic_model=False)  # TF-IDF ile benzerlik
    evaluator.evaluate()
