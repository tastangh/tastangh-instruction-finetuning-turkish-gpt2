import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from dataset_manager import DatasetManager
import os
import shutil
from google.colab import files  # Colab için gerekli

class Trainer:
    def __init__(self, model_name: str, output_dir: str):
        """
        Args:
            model_name (str): Modelin ismi veya dizini.
            output_dir (str): Eğitim çıktılarının kaydedileceği dizin.
        """
        self.model_name = model_name
        self.output_dir = output_dir

    def load_model_and_tokenizer(self):
        """
        Model ve tokenizer'ı yükler ve LoRA ayarlarını uygular.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # LoRA ayarları
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.1,
            bias="none",
            fan_in_fan_out=True,  # Conv1D için doğru yapılandırma
        )
        model = get_peft_model(model, lora_config)

        # Cihaz seçimi (GPU veya CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Tokenizer için pad_token kontrolü
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def fine_tune(self, model, tokenizer, train_dataset, validation_dataset):
        """
        Modeli verilen veri kümesi üzerinde ince ayar yapar.
        Args:
            model: Eğitim için kullanılacak model.
            tokenizer: Tokenizer nesnesi.
            train_dataset: Eğitim veri kümesi.
            validation_dataset: Doğrulama veri kümesi.
        """

        # Metrik hesaplama fonksiyonu
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = torch.argmax(torch.tensor(logits), dim=-1)
            labels = torch.tensor(labels)
            accuracy = (predictions == labels).float().mean().item()
            return {"accuracy": accuracy, "eval_loss": logits.mean().item()}

        # Eğitim argümanları
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,  # GPU hata çözümü için 1
            gradient_accumulation_steps=8,
            num_train_epochs=50,  # Daha kısa eğitim döngüleri
            save_total_limit=1,
            logging_dir=f"{self.output_dir}/logs",
            learning_rate=2e-5,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            max_grad_norm=1.0,
            warmup_steps=100,
            weight_decay=0.01,
            eval_strategy="steps",  # Her birkaç adımda bir değerlendirme yapılır
            eval_steps=500,
            load_best_model_at_end=True,  # En iyi modeli yüklemek için gerekli
        )

        # Early Stopping Callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,  # Erken durdurma için patience azaltıldı
            early_stopping_threshold=0.01,
        )

        # Trainer nesnesi
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,  
            tokenizer=tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,  # Metrik hesaplama fonksiyonu eklendi
            callbacks=[early_stopping],
        )

        print(f"Model {self.model_name} eğitiliyor...")
        trainer.train()
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print(f"Model {self.output_dir} dizinine kaydedildi.")

        zip_path = f"{self.output_dir}.zip"
        shutil.make_archive(self.output_dir, 'zip', self.output_dir)
        print(f"{zip_path} dosyası oluşturuldu. İndirme başlıyor...")
        files.download(zip_path)



if __name__ == "__main__":
    # Eğitimde kullanılacak veri kümeleri
    datasets = {
        "v1": "./dataset/v1.csv",
        # "v2": "./dataset/v2.csv",
        # "v3": "./dataset/v3.csv",
    }

    # Kullanılacak modeller
    models = [
        "ytu-ce-cosmos/turkish-gpt2-medium",
        # "ytu-ce-cosmos/turkish-gpt2-large",
    ]

    # Dataset yöneticisi
    dataset_manager = DatasetManager(datasets)

    # Her veri kümesi için döngü
    for dataset_name, dataset_path in datasets.items():
        # Veri kümesini yükle ve böl
        train_dataset, validation_dataset = dataset_manager.load_dataset(
            dataset_name, validation_size=0.2
        )

        # Her model için döngü
        for model_name in models:
            output_dir = f"./models/{model_name.split('/')[-1]}_{dataset_name}"
            trainer = Trainer(model_name, output_dir)

            # Model ve tokenizer yükle
            model, tokenizer = trainer.load_model_and_tokenizer()

            # Modeli ince ayar yaparak eğit
            trainer.fine_tune(model, tokenizer, train_dataset, validation_dataset)

    print("Tüm modeller eğitildi.")
