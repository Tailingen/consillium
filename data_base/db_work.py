import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
import hashlib
from llama_cpp import Llama

def initialize_models():
    """Инициализация всех моделей один раз перед созданием генераторов"""
    print("Инициализация моделей...")

    # 1. Инициализация BioBERT
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Загрузка BioBERT на {device}...")
    model_name = "dmis-lab/biobert-v1.1"
    biobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    biobert_model = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True, local_files_only=False).to(device)
    biobert_model.eval()

    # 2. Инициализация GGUF модели
    print("Загрузка медицинской Phi-3.5 Mini...")
    llm = Llama.from_pretrained(
        repo_id="mradermacher/JSL-Med-Phi-3.5-Mini-v3-i1-GGUF",
        filename="JSL-Med-Phi-3.5-Mini-v3.i1-Q4_K_M.gguf",
        n_ctx=512,
        n_gpu_layers=-1,
        n_threads=8, # Используйте больше ядер
        n_batch=2048,  # Увеличьте размер батча
        offload_kqv=True,  # Выгрузка слоёв на GPU
        verbose=False,
        logits_all=True
    )

    return biobert_tokenizer, biobert_model, llm

def qdrant_initialize():

    # 1.Конфигурация для локальной работы
    QDRANT_PATH = "./qdrant_data"  # Папка для хранения векторов

    # 2. Локальный Qdrant (без Docker)
    client = QdrantClient(path=QDRANT_PATH)  # Все данные сохраняются в папке

    # 3. Создаем коллекцию с оптимизированными параметрами
    COLLECTION_NAME = "medical_texts_local"

    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768,  # BioBERT размерность
                distance=Distance.COSINE
            ),
            optimizers_config={
                "memmap_threshold": 10000  # Оптимизация памяти
            }
        )
    print(f"Создана новая коллекция {COLLECTION_NAME}")
    return client

def upload_to_qdrant(client: QdrantClient, biobert_tokenizer, biobert_model):
    """Пакетная загрузка документов в Qdrant с автоматическим управлением ресурсами"""
    # Проверка входных параметров
    if client is None:
        raise ValueError("Qdrant client не инициализирован")
    if not hasattr(client, 'upsert'):
        raise ValueError("Некорректный Qdrant client")
    if biobert_tokenizer is None or biobert_model is None:
        raise ValueError("Модели BioBERT не инициализированы")

    # Конфигурация
    TEXTS_DIR = "C:\\papka\\python\\consillium\\files"
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COLLECTION_NAME = "medical_texts_local"

    try:
        # Проверка существования директории
        if not os.path.exists(TEXTS_DIR):
            raise FileNotFoundError(f"Директория {TEXTS_DIR} не найдена")

        def get_embedding(text: str) -> np.ndarray:
            """Генерация эмбеддинга для текста"""
            inputs = biobert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = biobert_model(**inputs)

            return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

        def process_files(directory: str):
            """Генератор для обработки файлов"""
            files = [f for f in os.listdir(directory) if f.endswith(".txt")]
            if not files:
                raise ValueError(f"В директории {directory} нет .txt файлов")

            for filename in tqdm(files, desc="Обработка файлов"):
                filepath = os.path.join(directory, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read().strip()

                    if not text:
                        continue

                    file_id = int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)

                    yield PointStruct(
                        id=file_id,
                        vector=get_embedding(text).tolist(),
                        payload={
                            "filename": filename,
                            "text_preview": text[:150] + "..." if len(text) > 150 else text,
                            "source": "local_processing"
                        }
                    )
                except Exception as e:
                    print(f"Ошибка обработки файла {filename}: {str(e)}")

        # Пакетная загрузка
        points_batch = []
        processed_count = 0
        success = False

        try:
            for point in process_files(TEXTS_DIR):
                points_batch.append(point)

                if len(points_batch) >= BATCH_SIZE:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points_batch,
                        wait=True
                    )
                    processed_count += len(points_batch)
                    points_batch = []
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

            if points_batch:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_batch
                )
                processed_count += len(points_batch)

            print(f"\nУспешно обработано {processed_count} файлов")
            success = True

        except Exception as e:
            print(f"Ошибка при загрузке данных: {str(e)}")
            success = False

        return success

    finally:
        # Гарантированное освобождение ресурсов
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        # Не закрываем client здесь, так как он передан извне
        # (закрытие должно быть на уровне вызывающего кода)


biobert_tokenizer, biobert_model, llm = initialize_models()
qdrant_client = qdrant_initialize()

try:
    # 1. Загрузка данных
    upload_to_qdrant(qdrant_client, biobert_tokenizer, biobert_model)

    # 2. Поиск/генерация (если нужно)
    # ...
finally:
    # Закрытие соединения при завершении работы
    qdrant_client.close()
    print("Соединение с Qdrant закрыто")