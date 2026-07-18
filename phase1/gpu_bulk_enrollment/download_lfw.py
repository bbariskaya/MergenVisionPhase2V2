import shutil
from pathlib import Path

TARGET = Path("/home/user/Workspace/MergenVisionPhase2v2/phase1/gpu_bulk_enrollment/lfw")
SOURCE = Path("/tmp/lfw-deepfunneled/lfw-deepfunneled")
CSV_SOURCE = Path("/tmp")

if not SOURCE.exists():
    raise FileNotFoundError(f"Kaynak bulunamadı: {SOURCE}")

# Hedefi tamamen temizle
if TARGET.exists():
    shutil.rmtree(TARGET)
TARGET.mkdir(parents=True, exist_ok=True)

# CSV dosyalarını kopyala
for csv_file in sorted(CSV_SOURCE.glob("lfw_*.csv")):
    shutil.copy2(csv_file, TARGET / csv_file.name)

# Kişi klasörlerini doğrudan lfw/ altına kopyala (wrapper yok)
for person_dir in sorted(SOURCE.iterdir()):
    if person_dir.is_dir():
        dest = TARGET / person_dir.name
        shutil.copytree(person_dir, dest)

print("Bitti. Hedef:", TARGET)
print("CSV dosyaları:", len(list(TARGET.glob("*.csv"))))
print("Kişi sayısı:", sum(1 for d in TARGET.iterdir() if d.is_dir()))
print("Toplam resim:", sum(1 for _ in TARGET.rglob("*.jpg")))
