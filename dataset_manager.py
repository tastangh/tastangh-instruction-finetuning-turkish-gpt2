import pandas as pd
from datasets import Dataset

class DatasetManager:
    def __init__(self, dataset_paths: dict):
        """
        Args:
            dataset_paths (dict): Veri kümesi adları ve dosya yollarını içeren sözlük.
        """
        self.dataset_paths = dataset_paths

    def load_dataset(self, dataset_name: str):
        """
        Belirtilen veri kümesini yükler ve Hugging Face Dataset formatına dönüştürür.
        """
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"{dataset_name} için bir dosya yolu belirtilmemiş.")

        # Veri kümesini yükle
        df = pd.read_csv(self.dataset_paths[dataset_name])
        print(f"{dataset_name} başarıyla yüklendi. Sütunlar: {df.columns.tolist()}")

        # Dataset formatlama
        dataset = Dataset.from_pandas(df)
        dataset = self.format_dataset(dataset, dataset_name)
        return dataset

    def format_dataset(self, dataset, dataset_name: str):
        """
        Veri kümesini verilen dataset_name'e uygun şekilde formatlar.

        Args:
            dataset: Hugging Face Dataset
            dataset_name (str): Dataset tipi (v1, v2, v3)
        
        Returns:
            Hugging Face Dataset: Formatlanmış veri kümesi
        """
        def format_function(example):
            if dataset_name == "v1":
                # V1: Soru ve İnsan Cevabı birleştirilir
                formatted_text = f"{example['soru']}\n{example['insan cevabı']}"
            elif dataset_name in ["v2", "v3"]:
                # V2 ve V3: Talimat, Giriş (varsa), Çıktı birleştirilir
                if "giriş" in example and example["giriş"]:
                    formatted_text = f"{example['talimat']}\n{example['giriş']}\n{example['çıktı']}"
                else:
                    formatted_text = f"{example['talimat']}\n{example['çıktı']}"
            return {"text": formatted_text}  # Yeni bir `text` alanı oluştur

        # Her örneğe formatlama işlevini uygula
        dataset = dataset.map(format_function)
        print(f"{dataset_name} veri kümesi başarıyla formatlandı.")
        return dataset