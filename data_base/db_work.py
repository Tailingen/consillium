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

biobert_tokenizer, biobert_model, llm = initialize_models()