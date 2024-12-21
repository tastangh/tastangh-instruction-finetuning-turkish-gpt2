import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments


def load_model_and_tokenizer(model_name: str):
    """
    Hugging Face'den model ve tokenizer yükler ve cihazına taşır.
    
    Args:
        model_name (str): Hugging Face model ismi.
    
    Returns:
        model: Yüklenen model.
        tokenizer: Yüklenen tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GPU kontrolü ve modele taşınması
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Model {device} üzerinde başarıyla yüklendi.")
    return model, tokenizer


def load_and_format_dataset(dataset_name: str):
    """
    Hugging Face'den bir veri seti yükler ve uygun formata dönüştürür.
    
    Args:
        dataset_name (str): Veri setinin Hugging Face adı.
    
    Returns:
        formatted_dataset: Formatlanmış veri seti.
    """
    dataset = load_dataset(dataset_name)

    # Veriyi formatlama
    def format_dataset(example):
        instruction = example.get("talimat", "")  # "talimat" sütunu komut içindir
        input_text = example.get("giriş", "")    # "giriş" sütunu input içindir
        target_text = example.get("çıktı", "")   # "çıktı" sütunu cevap içindir

        return {
            "instruction": instruction,
            "input": input_text,
            "output": target_text
        }

    formatted_dataset = dataset["train"].map(format_dataset)
    print("Veri seti başarıyla formatlandı.")
    return formatted_dataset


def preprocess_data(tokenizer, dataset):
    """
    Veri setini tokenizasyon yaparak modele uygun hale getirir.
    
    Args:
        tokenizer: Modelin tokenizer nesnesi.
        dataset: Formatlanmış veri seti.
    
    Returns:
        tokenized_dataset: Tokenize edilmiş veri seti.
    """
    def tokenize(example):
        inputs = f"{example['instruction']}\n{example['input']}\n"
        labels = example["output"]

        tokenized = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                labels,
                max_length=512,
                truncation=True,
                padding="max_length"
            )

        tokenized["labels"] = labels["input_ids"]
        return tokenized

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("Veri seti başarıyla tokenize edildi.")
    return tokenized_dataset


def configure_lora(model):
    """
    Model için LoRA yapılandırmasını uygular.
    
    Args:
        model: Orijinal Hugging Face modeli.
    
    Returns:
        model: LoRA ile sarmalanmış model.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA yapılandırması başarıyla tamamlandı.")
    return model


def train_model(model, tokenizer, train_dataset):
    """
    Modeli verilen veri seti üzerinde eğitir.
    
    Args:
        model: Eğitime hazır model.
        tokenizer: Modelin tokenizer nesnesi.
        train_dataset: Tokenize edilmiş eğitim veri seti.
    
    Returns:
        None
    """
    training_args = TrainingArguments(
        output_dir="./turkish-gpt2-medium-finetuned",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        learning_rate=5e-5,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    print("Model eğitimi tamamlandı.")


def test_model(model, tokenizer):
    """
    Eğitilen modeli test etmek için bir örnek çalıştırır.
    
    Args:
        model: Eğitilen model.
        tokenizer: Modelin tokenizer nesnesi.
    
    Returns:
        None
    """
    test_instruction = "Verilen metni kısa bir şekilde özetleyin."
    test_input = "Bugün hava çok güzel ve insanlar parklara akın etti."

    inputs = tokenizer(f"{test_instruction}\n{test_input}\n", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50)

    print("Modelin çıktısı:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    # Model ve Tokenizer Yükleme
    model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Veri Seti Hazırlama
    dataset_name = "merve/turkish_instructions"
    formatted_dataset = load_and_format_dataset(dataset_name)

    # Veri Seti Tokenizasyonu
    train_dataset = preprocess_data(tokenizer, formatted_dataset)

    # LoRA Yapılandırması
    model = configure_lora(model)

    # Model Eğitimi
    train_model(model, tokenizer, train_dataset)

    # Model Testi
    test_model(model, tokenizer)
