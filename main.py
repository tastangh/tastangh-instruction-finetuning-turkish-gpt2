from model_trainer import ModelTrainer

if __name__ == "__main__":
    # Kullanılacak veri kümeleri ve modeller
    datasets = [
        "./dataset/v1.csv",
        "./dataset/v2.csv",
        "./dataset/v3.csv",
    ]
    models = [
        "ytu-ce-cosmos/turkish-gpt2-medium",
        "ytu-ce-cosmos/turkish-gpt2-large",
    ]

    # Her model ve veri kümesi kombinasyonu için fine-tune işlemi
    for dataset in datasets:
        for model_name in models:
            # Çıktı dizinini model ve veri kümesine göre isimlendir
            output_dir = f"./fine_tuned_models/{model_name.split('/')[-1]}_{dataset.split('/')[-1].split('.')[0]}"

            # ModelTrainer örneği oluştur
            trainer = ModelTrainer(model_name, output_dir)

            # Model ve tokenizer'ı yükle
            model, tokenizer = trainer.load_model_and_tokenizer()

            # Fine-tune işlemi
            trainer.fine_tune(model, tokenizer, dataset)

            # Test işlemi
            trainer.test_model(
                model,
                tokenizer,
                "Bir metni özetle.",
                "Bugün hava çok güzel. İnsanlar dışarıda yürüyüş yapıyor."
            )
