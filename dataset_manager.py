import pandas as pd
from datasets import Dataset

class DatasetManager:
    def __init__(self, dataset_paths: dict):
        """
        Args:
            dataset_paths (dict): Veri kümesi adları ve dosya yollarını içeren sözlük.
        """
        self.dataset_paths = dataset_paths

    def load_dataset(self, dataset_name: str, validation_size: float = 0.2):
        """
        Belirtilen veri kümesini yükler, Hugging Face Dataset formatına dönüştürür
        ve train-validation olarak ayırır.

        Args:
            dataset_name (str): Veri kümesi adı.
            validation_size (float): Validation set oranı (default: 0.2).

        Returns:
            tuple: (train_dataset, validation_dataset)
        """
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"{dataset_name} için bir dosya yolu belirtilmemiş.")

        # Veri kümesini yükle
        df = pd.read_csv(self.dataset_paths[dataset_name])
        print(f"{dataset_name} başarıyla yüklendi. Sütunlar: {df.columns.tolist()}")

        # Dataset formatlama
        dataset = Dataset.from_pandas(df)
        dataset = self.format_dataset(dataset, dataset_name)

        # Train-validation olarak ayırma
        dataset_split = dataset.train_test_split(test_size=validation_size)
        train_dataset = dataset_split["train"]
        validation_dataset = dataset_split["test"]

        print(f"{dataset_name} veri kümesi başarıyla bölündü:")
        print(f"  Train set boyutu: {len(train_dataset)}")
        print(f"  Validation set boyutu: {len(validation_dataset)}")

        return train_dataset, validation_dataset

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
