from qdrant_client import QdrantClient
import torch
from functools import lru_cache

class MedicalTextGenerator:
    def __init__(self, biobert_tokenizer, biobert_model, llm, qdrant_path: str = "./qdrant_data"):
        """
        Автономный генератор с управлением подключением к Qdrant

        Args:
            biobert_tokenizer: инициализированный токенизатор BioBERT
            biobert_model: загруженная модель BioBERT
            llm: инициализированная GGUF модель
            qdrant_path: путь к данным Qdrant
        """
        self.biobert_tokenizer = biobert_tokenizer
        self.biobert_model = biobert_model
        self.llm = llm
        self.qdrant_path = qdrant_path
        self.client = None  # Будет инициализирован при входе в контекст

    def __enter__(self):
        """Открываем соединение при входе в контекст"""
        self.client = QdrantClient(
            path=self.qdrant_path,
            force_disable_lock=True,
            prefer_grpc=True
        )
        self._verify_collection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрываем соединение при выходе из контекста"""
        self.close()

    def close(self):
        """Явное закрытие соединения"""
        if self.client is not None:
            self.client.close()
            self.client = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _verify_collection(self):
        """Проверка коллекции"""
        try:
            collection_info = self.client.get_collection("medical_texts_local")
            if collection_info.config.params.vectors.size != 768:
                raise ValueError("Несоответствие размерности векторов (ожидается 768)")
        except Exception as e:
            raise RuntimeError(f"Ошибка проверки коллекции: {e}")

    def get_embedding_query(self, text: str) -> list[float]:
        """Генерация эмбеддинга с BioBERT"""
        inputs = self.biobert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(next(self.biobert_model.parameters()).device)

        with torch.no_grad():
            outputs = self.biobert_model(**inputs)

        # Усреднение с учетом маски внимания
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return embeddings.cpu().numpy().squeeze().tolist()

    def search_relevant_documents(self, query: str, top_k: int = 3):
        """Поиск документов с использованием BioBERT"""
        try:
            query_vector = self.get_embedding_query(query)
            results = self.client.search(
                collection_name="medical_texts_local",
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=0.4
            )
            return [hit.payload.get('text', '') for hit in results]
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []

    @lru_cache(maxsize=100)  # Кэширует последние 100 запросов
    def generate_medical_text(self, query: str, max_tokens: int = 512) -> str:
        """Генерация медицинского ответа"""
        try:
            context_docs = self.search_relevant_documents(query)
            if not context_docs:
                return "No relevant documents found" # Четкое уведомление об отсутствии данных

            context = "\n".join([f"[Источник {i+1}]: {doc[:300]}"
                              for i, doc in enumerate(context_docs)])[:1500]

            prompt = f"""<|system|>
You are a professor of oncology. Give a detailed answer to the question. Answer in English only, do not use Russian
Answer ONLY based on the provided medical guidelines.
If the information is not in the documents, say: Not in my database:</|system|>

<|user|>
Вопрос: {query}

Контекст:
{context}
</|user|>

<|assistant|>"""

            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3, # Низкая "креативность"
                top_p=0.9,
                repeat_penalty=1.1,
                echo=False
            )

            return response['choices'][0]['text'].strip()
        except Exception as e:
            return f"Ошибка генерации: {str(e)}"

    def close(self):
        """Освобождение ресурсов"""
        if hasattr(self, 'client'):
            self.client.close()