import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import nltk

# — Загрузка токенизатора предложений
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
# — Загрузка модели BioBERT
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# — Устройство: GPU или CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

async def vectorize(file):

    # — Векторизация одного предложения
    def embed_sentence(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    # — Векторизация одного текста (по предложениям)
    def get_embedding(text):
        sentences = sent_tokenize(text)
        sentence_embeddings = [embed_sentence(sent) for sent in sentences]
        return np.mean(sentence_embeddings, axis=0)

    # — Загрузка строк из файла
    with open(file, "r", encoding="utf-8") as f: # заменить на свой файл
        texts = [line.strip() for line in f if line.strip()]

    # — Векторизация всех строк
    embeddings = [get_embedding(text) for text in texts]
    # === Сохранение ===
    np.save("embeddings/embeddings_from_txt.npy", embeddings)
    np.savetxt("embeddings/embeddings_from_txt.csv", embeddings, delimiter=",")
    print("✅ Векторизация завершена.")
    return {'detail': {'code': 'OK', 'message': 'Векторизация текста прошла успешно'}}

# model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Или другая модель
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
#
# def vectorize_text(text: str):
#     # Токенизация текста
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#
#     # Получение векторного представления
#     with torch.no_grad():
#         embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Усреднение по токенам для получения одного вектора
#
#     return embeddings.numpy()  # Возвращаем вектор как NumPy массив
#
# def vectorize(file):
#     with open(file, "r", encoding="utf-8") as f:
#         text = f.read()
#     result = vectorize_text(text)
#     return result