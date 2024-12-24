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
    def __init__(self, model_name: str, output_dir: str, log_file: str):
        self.model_name = model_name
        self.output_dir = output_dir
        setup_logging(log_file)

    def load_model_and_tokenizer(self):
        logging.info(f"Model ve tokenizer yükleniyor: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Geliştirilmiş LoRA Konfigürasyonu
        lora_config = LoraConfig(
            r=16,  # Adaptasyon matrisi boyutu
            lora_alpha=32,  # Etkinlik artırıcı ölçekleme
            target_modules=[
                "q_proj", "k_proj", "v_proj",  # Ayrıştırılmış dikkat modülleri
                "c_attn",  # Birleşik dikkat modülü
                "o_proj",  # Dikkat çıktı projeksiyonu
                "ffn_up_proj", "ffn_down_proj",  # Feedforward projeksiyonları
                "embed_tokens", "embed_positions",  # Gömülü temsiller
                "norm"  # Normalizasyon
            ],
            lora_dropout=0.1,  # Aşırı öğrenmeyi önlemek için dropout
            bias="none",  # Eklenen biasları eğitme
            task_type="CAUSAL_LM",  # Nedensel dil modelleme
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

        # bf16 veya fp16 otomatik seçimi
        use_bf16 = torch.cuda.is_bf16_supported()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,  # GPU bellek sınırına göre ayarlanabilir
            gradient_accumulation_steps=8,  # Daha büyük efektif batch size
            num_train_epochs=1,  # Daha kısa süreli ince ayar için 1 epoch
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,  # Daha sık loglama
            learning_rate=1e-4,  # Daha yüksek başlangıç öğrenme oranı
            warmup_steps=100,  
            weight_decay=0.01,
            bf16=use_bf16,  # bf16 destekliyorsa kullan
            fp16=not use_bf16,  # Aksi durumda fp16 kullan
            save_strategy="no",  # Eğitim sırasında model kaydedilmez
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            args=training_args,
            callbacks=[TrainingLoggerCallback(logging_steps=100)],  # Eğitim log callback'i
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
