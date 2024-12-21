from datasets import load_dataset
import pandas as pd

def save_merve_dataset_to_csv(output_path: str):
    """
    Hugging Face'den "merve/turkish_instructions" veri kümesini indirir ve CSV olarak kaydeder.
    
    Args:
        output_path (str): CSV dosyasının kaydedileceği yol (örn: "./merve_instructions.csv").
    
    Returns:
        None
    """
    # Merve veri kümesini indir
    dataset = load_dataset("merve/turkish_instructions")
    
    # Eğitim verisini al
    train_data = dataset["train"]
    
    # Hugging Face Dataset'i pandas DataFrame'e dönüştür
    df = pd.DataFrame(train_data)

    # Sütunları düzenle (isteğe bağlı, Türkçe sütun isimlerini koruyabilirsiniz)
    df = df.rename(columns={
        "talimat": "instruction",
        "giriş": "input",
        "çıktı": "output"
    })

    # DataFrame'i CSV olarak kaydet
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Veri kümesi başarıyla {output_path} dosyasına kaydedildi.")

if __name__ == "__main__":
    # CSV dosyasını kaydetmek istediğiniz yolu belirtin
    output_csv_path = "./v2.csv"
    
    # Kaydetme işlemini çalıştır
    save_merve_dataset_to_csv(output_csv_path)
