import pandas as pd
from datasets import Dataset, DatasetDict

class DatasetManager:
    def __init__(self, dataset_paths: dict):
        """
        DatasetManager sınıfı, veri kümelerini yüklemek ve işlemek için kullanılır.

        Args:
            dataset_paths (dict): Veri kümesi adları ve dosya yollarını içeren sözlük.
        """
        self.dataset_paths = dataset_paths

    def load_dataset(self, dataset_name: str, test_size: float = 0.2, validation_size: float = 0.2):
        """
        Veri kümesini yükler ve train-validation-test olarak böler.

        Args:
            dataset_name (str): Veri kümesi adı.
            test_size (float): Test set oranı (default: 0.2).
            validation_size (float): Validation set oranı (default: 0.2).

        Returns:
            tuple: (train_dataset, validation_dataset, test_dataset)
        """
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"{dataset_name} için bir dosya yolu belirtilmemiş.")
        
        if test_size + validation_size >= 1.0:
            raise ValueError("Test ve Validation oranlarının toplamı 1'den küçük olmalıdır.")

        # Veri kümesini yükle
        df = pd.read_csv(self.dataset_paths[dataset_name])
        print(f"{dataset_name} başarıyla yüklendi. Toplam satır sayısı: {len(df)}")

        # Dataset formatlama
        dataset = Dataset.from_pandas(df)
        dataset = self.format_dataset(dataset, dataset_name)

        # Dataset'i test setine ayır
        dataset_split = dataset.train_test_split(test_size=test_size)
        test_dataset = dataset_split["test"]
        train_dataset = dataset_split["train"]

        # Eğitim kümesini validation setine ayır
        train_validation_split = train_dataset.train_test_split(test_size=validation_size)
        train_dataset = train_validation_split["train"]
        validation_dataset = train_validation_split["test"]

        print(f"{dataset_name} veri kümesi başarıyla bölündü:")
        print(f"  Train set boyutu: {len(train_dataset)}")
        print(f"  Validation set boyutu: {len(validation_dataset)}")
        print(f"  Test set boyutu: {len(test_dataset)}")

        return train_dataset, validation_dataset, test_dataset

    def format_dataset(self, dataset: Dataset, dataset_name: str) -> Dataset:
        """
        Veri kümesini verilen dataset_name'e uygun şekilde formatlar.

        Args:
            dataset: Hugging Face Dataset.
            dataset_name (str): Dataset tipi (v1, v2, v3).

        Returns:
            Hugging Face Dataset: Formatlanmış veri kümesi.
        """
        def format_function(example):
            if dataset_name == "v1":
                # V1: Soru ve İnsan Cevabı birleştirilir
                formatted_text = f"{example['soru']}\n{example['insan cevabı']}"
            else:
                # Talimat, Giriş ve Çıktı birleştirilir
                if "giriş" in example and example["giriş"]:
                    formatted_text = f"{example['talimat']}\n{example['giriş']}\n{example['çıktı']}"
                else:
                    formatted_text = f"{example['talimat']}\n{example['çıktı']}"
            return {"text": formatted_text}

        # Formatlama işlemini tüm verilere uygula
        dataset = dataset.map(format_function)
        print(f"{dataset_name} veri kümesi başarıyla formatlandı.")
        return dataset
