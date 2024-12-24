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
    """Logları hem terminale hem dosyaya yönlendirme."""
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
        """Belirli adımlarda log metrikleri yazdır."""
        if state.global_step % self.logging_steps == 0 and state.log_history:
            logs = state.log_history[-1]
            metrics = {k: v for k, v in logs.items() if k in ["loss", "grad_norm", "learning_rate", "epoch"]}
            logging.info(f"Training Logs at step {state.global_step}: {metrics}")


class Trainer:
    def __init__(self, model_name: str, output_dir: str, log_file: str, lora_params: dict):
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_params = lora_params
        setup_logging(log_file)

    def load_model_and_tokenizer(self):
        logging.info(f"Model ve tokenizer yükleniyor: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # LoRA Konfigürasyonu
        lora_config = LoraConfig(
            r=self.lora_params.get("r", 32),
            lora_alpha=self.lora_params.get("lora_alpha", 64),
            target_modules=self.lora_params.get("target_modules", ["c_attn", "c_proj", "q_attn", "v_proj"]),
            lora_dropout=self.lora_params.get("lora_dropout", 0.1),
            bias=self.lora_params.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Cihaz Ayarı
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Pad token kontrolü
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logging.info(f"Model ve tokenizer başarıyla yüklendi: {self.model_name}")
        return model, tokenizer

    def fine_tune(self, model, tokenizer, train_dataset):
        logging.info(f"Model {self.model_name} ince ayar işlemi başlatılıyor.")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,  # Colab İçin Düşük Batch Size
            gradient_accumulation_steps=8,  # Etkin Batch Size Artırır
            num_train_epochs=5,  # Hızlı Deneme Eğitimi
            learning_rate=5e-5,  # Daha Stabil Öğrenme Oranı
            warmup_steps=100,  # Isınma Adımları
            weight_decay=0.01,
            fp16=True,  # Daha Az Bellek Kullanımı için FP16
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=500,  # Daha Sık Loglama
            save_strategy="epoch",  # Her Epoch’ta Kaydetme
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            args=training_args,
            callbacks=[TrainingLoggerCallback(logging_steps=500)],  # Eğitim log callback'i
        )

        trainer.train()
        logging.info(f"Model {self.model_name} ince ayar işlemi başarıyla tamamlandı.")

        logging.info(f"Model ve tokenizer kaydediliyor: {self.output_dir}")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        shutil.make_archive(self.output_dir, 'zip', self.output_dir)
        logging.info(f"Model {self.output_dir} dizinine başarıyla kaydedildi.")


def train_model(model_name, dataset_name, dataset_path, lora_params):
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
    trainer = Trainer(model_name, output_dir, log_file, lora_params)

    # Model ve tokenizer yükle
    model, tokenizer = trainer.load_model_and_tokenizer()

    # Modeli ince ayar yaparak eğit
    trainer.fine_tune(model, tokenizer, train_dataset)

    logging.info(f"[SUCCESS] Eğitim tamamlandı: Model={model_name}, Dataset={dataset_name}")


if __name__ == "__main__":
    # Eğitimde kullanılacak veri kümeleri ve modeller 
    datasets = {
        "v3": "./dataset/v3.csv",
        "v2": "./dataset/v2.csv",
        "v1": "./dataset/v1.csv",
    }

    models = [
        "ytu-ce-cosmos/turkish-gpt2-large",
        "ytu-ce-cosmos/turkish-gpt2-medium",
    ]

    lora_params = {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["c_attn", "c_proj", "q_attn", "v_proj"], 
        "bias": "none",
    }

    for dataset_name, dataset_path in datasets.items():
        for model_name in models:
            train_model(model_name, dataset_name, dataset_path, lora_params)
