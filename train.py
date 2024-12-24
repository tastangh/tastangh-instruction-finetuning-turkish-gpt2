import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from dataset_manager import DatasetManager
import os
import shutil
import logging


# Logging Yapılandırması
def setup_logging(log_file):
    # Loglama ayarları
    logging.basicConfig(
        filename=log_file,
        filemode="w",  # Her çalıştırmada log dosyasını sıfırlar
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console_handler = logging.StreamHandler()  # Konsol için handler
    console_handler.setLevel(logging.INFO)  # Konsol seviyesini belirle
    console_handler.setFormatter(logging.Formatter("%(message)s"))  # Sadece mesajı göster
    logging.getLogger().addHandler(console_handler)  # Konsola yazdırmayı etkinleştir


# Custom Callback for Training Logs
class TrainingLoggerCallback(TrainerCallback):
    def __init__(self, logging_steps):
        super().__init__()
        self.logging_steps = logging_steps

    def on_step_end(self, args, state, control, **kwargs):
        """Log metrikleri belirli adımlarda yazdır."""
        if state.global_step % self.logging_steps == 0:  # logging_steps sıklığında kontrol
            logs = state.log_history[-1] if state.log_history else {}
            metrics = {k: v for k, v in logs.items() if k in ["loss", "grad_norm", "learning_rate", "epoch"]}
            logging.info(f"Training Logs at step {state.global_step}: {metrics}")


class Trainer:
    def __init__(self, model_name: str, output_dir: str, log_file: str):
        self.model_name = model_name
        self.output_dir = output_dir
        setup_logging(log_file)

    def load_model_and_tokenizer(self):
        logging.info(f"Model ve tokenizer yükleniyor: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        lora_config = LoraConfig(
            r=32,  # Daha yüksek kapasite için
            lora_alpha=64,  # Modelin öğrenme kapasitesini artırır
            target_modules=["c_attn", "c_proj", "q_attn", "v_proj"],  # Dikkat mekanizması katmanları
            lora_dropout=0.1,  # Aşırı öğrenmeyi engellemek için düşük dropout
            bias="none",  # Bias terimlerini dondur
        )

        model = get_peft_model(model, lora_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logging.info(f"Model ve tokenizer başarıyla yüklendi: {self.model_name}")
        return model, tokenizer

    def fine_tune(self, model, tokenizer, train_dataset):
        logging.info(f"Model {self.model_name} ince ayar işlemi başlatılıyor.")
        logging_steps = 10  # Konsola yazıldığı sıklık
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,  # GPU bellek sınırına göre ayarlanabilir 
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=logging_steps,
            learning_rate=1e-5,
            warmup_steps=500,
            weight_decay=0.01,
            fp16=True,
            save_total_limit=1,  # Sadece en son modeli sakla
            save_steps=None,  # Adımlarda kaydetmeyi devre dışı bırak
            save_strategy="no",  # Eğitim sırasında hiç kaydetme
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            args=training_args,
            callbacks=[TrainingLoggerCallback(logging_steps)],  # Eğitim log callback'i eklendi
        )

        trainer.train()
        logging.info(f"Model {self.model_name} ince ayar işlemi başarıyla tamamlandı.")

        logging.info(f"Model ve tokenizer kaydediliyor: {self.output_dir}")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        shutil.make_archive(self.output_dir, 'zip', self.output_dir)
        logging.info(f"Model {self.output_dir} dizinine başarıyla kaydedildi.")


def train_model(model_name, dataset_name, dataset_path):
    """
    Her model-dataset kombinasyonu için eğitim işlemini yürütür.
    """
    log_file = f"logs/{model_name.split('/')[-1]}_{dataset_name}.log"
    os.makedirs("logs", exist_ok=True)  # logs dizinini oluştur
    setup_logging(log_file)  # Loglama yapılandırmasını başlat
    logging.info(f"[INFO] Eğitim başlıyor: Model={model_name}, Dataset={dataset_name}")

    # Dataset yöneticisi ve yükleme
    dataset_manager = DatasetManager({dataset_name: dataset_path})
    train_dataset = dataset_manager.load_dataset(dataset_name)
    logging.info(f"Dataset başarıyla yüklendi: {dataset_name}")

    # Çıktı dizinini oluştur
    output_dir = f"./models/{model_name.split('/')[-1]}_{dataset_name}"
    trainer = Trainer(model_name, output_dir, log_file)

    # Model ve tokenizer yükle
    model, tokenizer = trainer.load_model_and_tokenizer()

    # Modeli ince ayar yaparak eğit
    trainer.fine_tune(model, tokenizer, train_dataset)

    logging.info(f"[SUCCESS] Eğitim tamamlandı: Model={model_name}, Dataset={dataset_name}")


if __name__ == "__main__":
    # Eğitimde kullanılacak veri kümeleri ve modeller
    datasets = {
        "v1": "./dataset/v1.csv",
        "v2": "./dataset/v2.csv",
        "v3": "./dataset/v3.csv",
    }

    models = [
        "ytu-ce-cosmos/turkish-gpt2-medium",
        "ytu-ce-cosmos/turkish-gpt2-large",
    ]

    for dataset_name, dataset_path in datasets.items():
        for model_name in models:
            train_model(model_name, dataset_name, dataset_path)
