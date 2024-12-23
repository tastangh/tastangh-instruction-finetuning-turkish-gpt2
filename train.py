import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from dataset_manager import DatasetManager


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
            r=16,  # Daha yüksek rank değeri
            lora_alpha=32,  # Daha güçlü ağırlık faktörü
            target_modules=["c_attn", "c_proj"],  # Önemli transformer katmanları
            lora_dropout=0.1,  # Aşırı öğrenmeyi önlemek için
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        # Cihaz seçimi (GPU veya CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Tokenizer için pad_token kontrolü
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def fine_tune(self, model, tokenizer, train_dataset):
        """
        Modeli verilen veri kümesi üzerinde ince ayar yapar.
        """
        # Eğitim argümanları
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,  # gpu error gidermek için 2 ye indirdim
            gradient_accumulation_steps=8,  # Gradyan biriktirme
            num_train_epochs=100,  # Daha uzun eğitim döngüleri
            save_steps=1000,  # Her 1000 adımda bir kontrol noktası kaydedilir
            evaluation_strategy="steps",  # Değerlendirme her belirli adımda yapılır
            eval_steps=1000,  # Değerlendirme sıklığı save_steps ile uyumlu yapılır
            save_total_limit=1,  # Yalnızca en iyi modeli sakla
            load_best_model_at_end=True,  # Eğitim sonunda en iyi modeli yükle
            logging_dir=f"{self.output_dir}/logs",
            learning_rate=2e-5,  # Daha küçük öğrenme oranı
            bf16=torch.cuda.is_bf16_supported(),  # GPU BF16 destekliyorsa kullanılır
            fp16=not torch.cuda.is_bf16_supported(),  # Aksi durumda FP16 kullanılır
            max_grad_norm=1.0,  # Gradyan normu kesimi
            warmup_steps=100,  # Öğrenme oranını sabitlemek için ısınma adımları
            weight_decay=0.01,  # Overfitting'i azaltmak için ağırlık sönümleme
        )

        # Early Stopping Callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=5,  # 5 değerlendirme süresince iyileşme olmazsa durur
            early_stopping_threshold=0.01,  # Minimum iyileşme miktarı (opsiyonel)
        )
        
        # Trainer nesnesi
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            args=training_args,
        )

        print(f"Model {self.model_name} eğitiliyor...")
        trainer.train()
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print(f"Model {self.output_dir} dizinine kaydedildi.")


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
        # Veri kümesini yükle ve işle
        train_dataset = dataset_manager.load_dataset(dataset_name)

        # Her model için döngü
        for model_name in models:
            # Çıktı dizinini oluştur
            output_dir = f"./models/{model_name.split('/')[-1]}_{dataset_name}"
            trainer = Trainer(model_name, output_dir)

            # Model ve tokenizer yükle
            model, tokenizer = trainer.load_model_and_tokenizer()

            # Modeli ince ayar yaparak eğit
            trainer.fine_tune(model, tokenizer, train_dataset)
