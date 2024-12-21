import pandas as pd
from datasets import Dataset

def create_v3_dataset(v1_path: str, v2_path: str, v3_output_path: str):
    """
    V1 ve V2 veri kümelerini birleştirerek V3 veri kümesini oluşturur.
    
    Args:
        v1_path (str): V1 veri kümesinin CSV dosya yolu.
        v2_path (str): V2 veri kümesinin CSV dosya yolu.
        v3_output_path (str): V3 için oluşturulacak CSV dosyasının yolu.
    
    Returns:
        Dataset: Hugging Face Dataset formatında birleştirilmiş veri kümesi.
    """
    # V1 veri kümesini yükle
    v1 = pd.read_csv(v1_path)
    print("V1 sütunları:", v1.columns)
    v1 = v1.rename(columns={
        "soru": "talimat",
        "insan cevabı": "çıktı"
    })
    v1["giriş"] = ""  # V1'de giriş sütunu yok, boş bırakıyoruz

    # V2 veri kümesini yükle ve sütun isimlerini düzelt
    v2 = pd.read_csv(v2_path)
    print("V2 sütunları (orijinal):", v2.columns)
    v2.columns = v2.columns.str.strip()  # Fazladan boşlukları kaldır
    print("V2 sütunları (düzeltilmiş):", v2.columns)

    v2 = v2.rename(columns={
        "instruction": "talimat",
        "giriş": "giriş",
        "çıktı": "çıktı"
    })

    # V1 ve V2'yi birleştir
    combined = pd.concat([v1[["talimat", "giriş", "çıktı"]], 
                          v2[["talimat", "giriş", "çıktı"]]])

    # Birleştirilen veri kümesini CSV olarak kaydet
    combined.to_csv(v3_output_path, index=False, encoding="utf-8")
    print(f"V3 veri kümesi başarıyla {v3_output_path} dosyasına kaydedildi.")

    # V3'ün V1 ve V2'yi içerip içermediğini kontrol et
    check_v3_contains_v1_v2(v1, v2, combined)

    # Hugging Face Dataset formatına dönüştür ve döndür
    return Dataset.from_pandas(combined)

def check_v3_contains_v1_v2(v1: pd.DataFrame, v2: pd.DataFrame, v3: pd.DataFrame):
    """
    V3'ün V1 ve V2'yi içerip içermediğini kontrol eder.
    
    Args:
        v1 (pd.DataFrame): V1 veri kümesi.
        v2 (pd.DataFrame): V2 veri kümesi.
        v3 (pd.DataFrame): V3 veri kümesi.
    
    Returns:
        None
    """
    # V1'in V3'te olup olmadığını kontrol et
    v1_missing = v1[~v1[["talimat", "giriş", "çıktı"]].apply(tuple, axis=1).isin(v3.apply(tuple, axis=1))]
    if not v1_missing.empty:
        print("V3'te eksik olan V1 verileri:")
        print(v1_missing)
    else:
        print("V3, V1'in tüm verilerini içeriyor.")

    # V2'nin V3'te olup olmadığını kontrol et
    v2_missing = v2[~v2[["talimat", "giriş", "çıktı"]].apply(tuple, axis=1).isin(v3.apply(tuple, axis=1))]
    if not v2_missing.empty:
        print("V3'te eksik olan V2 verileri:")
        print(v2_missing)
    else:
        print("V3, V2'nin tüm verilerini içeriyor.")

if __name__ == "__main__":
    # V1, V2 ve V3 dosya yollarını belirtin
    v1_path = "../dataset/v1.csv"  # V1 veri kümesinin CSV dosyası
    v2_path = "../dataset/v2.csv"  # V2 veri kümesinin CSV dosyası
    v3_output_path = "../dataset/v3.csv"  # V3 için oluşturulacak CSV dosyası

    # V3 veri kümesini oluştur
    v3_dataset = create_v3_dataset(v1_path, v2_path, v3_output_path)
