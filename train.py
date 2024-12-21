import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from dataset_manager import DatasetManager


class Trainer:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir

    def load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Belleği optimize etmek için LoRA kullanımı
        lora_config = LoraConfig(
            r=4,  # Rank değerini düşük tutarak bellek kullanımını azaltın
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],  # Belirli katmanları optimize edin
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        # GPU veya CPU seçimi
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Tokenizer için pad_token ekleme
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def fine_tune(self, model, tokenizer, train_dataset):
        """
        Modeli verilen veri kümesinde ince ayar yapar.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,  # Küçük batch boyutu
            gradient_accumulation_steps=4,  # Gradyan biriktirme
            num_train_epochs=3,
            save_steps=500,
            logging_dir=f"{self.output_dir}/logs",
            learning_rate=5e-5,
            bf16=torch.cuda.is_bf16_supported(),  # GPU BF16 destekliyorsa kullanılır
            fp16=not torch.cuda.is_bf16_supported(),  # Aksi durumda FP16 kullanılır
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            args=training_args,
        )

        print(f"Model {self.model_name} eğitiliyor...")
        for step in range(len(train_dataset)):
            # Eğitim adımlarında bellek temizliği
            if step % 100 == 0:
                torch.cuda.empty_cache()

        trainer.train()
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print(f"Model {self.output_dir} dizinine kaydedildi.")


if __name__ == "__main__":
    datasets = {
        "v1": "./dataset/v1.csv",
        "v2": "./dataset/v2.csv",
        "v3": "./dataset/v3.csv",
    }
    models = [
        "ytu-ce-cosmos/turkish-gpt2-medium",
        "ytu-ce-cosmos/turkish-gpt2-large",
    ]

    # Dataset yöneticisi
    dataset_manager = DatasetManager(datasets)

    for dataset_name, dataset_path in datasets.items():
        # Veri kümesini yükle ve formatla
        train_dataset = dataset_manager.load_dataset(dataset_name)
        for model_name in models:
            # Çıktı dizinini oluştur
            output_dir = f"./models/{model_name.split('/')[-1]}_{dataset_name}"
            trainer = Trainer(model_name, output_dir)

            # Model ve tokenizer yükle
            model, tokenizer = trainer.load_model_and_tokenizer()

            # Modeli ince ayar yaparak eğit
            trainer.fine_tune(model, tokenizer, train_dataset)
