import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Evaluator:
    def __init__(self, model_dirs: list, test_dataset_path: str, output_csv: str):
        """
        Args:
            model_dirs (list): Tüm modellerin bulunduğu dizinlerin listesi.
            test_dataset_path (str): Test veri kümesi dosya yolu.
            output_csv (str): Çıktıların kaydedileceği CSV dosya yolu.
        """
        self.model_dirs = model_dirs
        self.test_dataset_path = test_dataset_path
        self.output_csv = output_csv

    def load_model_and_tokenizer(self, model_dir):
        """
        Model ve tokenizer'ı verilen dizinden yükler.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer

    def evaluate(self):
        """
        Tüm modelleri test eder ve sonuçları tek bir tabloya yazar.
        """
        # Test veri kümesini yükle
        test_data = pd.read_csv(self.test_dataset_path)
        results = test_data.copy()

        # Her modeli sırayla test et
        for idx, model_dir in enumerate(self.model_dirs, start=1):
            model, tokenizer = self.load_model_and_tokenizer(model_dir)
            model_column_name = f"model{idx} çıktısı"
            model_outputs = []

            print(f"Model {model_dir} test ediliyor...")
            for _, row in test_data.iterrows():
                question = row["soru"]
                inputs = tokenizer(f"{question}\n", return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_length=50)
                model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                model_outputs.append(model_output)

            results[model_column_name] = model_outputs
            print(f"Model {model_dir} test işlemi tamamlandı.")

        # Sonuçları kaydet
        results.to_csv(self.output_csv, index=False, encoding="utf-8")
        print(f"Sonuçlar {self.output_csv} dosyasına kaydedildi.")


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
    output_csv = "./outputs/test_results.csv"

    # Değerlendirme işlemini başlat
    evaluator = Evaluator(models, test_dataset_path, output_csv)
    evaluator.evaluate()
