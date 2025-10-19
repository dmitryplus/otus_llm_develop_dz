import json
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_community.llms import YandexGPT
from langchain.chains import LLMChain
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall
from datasets import Dataset

# Импортируем вспомогательные функции
from src.utils import (
    load_base_questions,
)


YC_FOLDER_ID = os.getenv("folder_id")
YC_API_KEY = os.getenv("ya_token")


class RAGEvaluator:
    """
    Класс для оценки RAG системы с использованием RAGAS метрик
    """

    def __init__(self, folder_id: str, api_key: str):
        """Инициализация базовых параметров"""
        self.folder_id = folder_id
        self.api_key = api_key
        self.yandex_embeddings = YandexGPTEmbeddings(
            model="doc",
            folder_id=folder_id,
            iam_token=api_key
        )

        # Путь к файлу с полным текстом вопросов
        QUESTIONS_FILE_WITH_SMALL_RESPONSE = "tests/goldens_small_response.json"

        # Загружаем вопросы из файла (или используем существующий словарь)
        self.questions = load_base_questions(QUESTIONS_FILE_WITH_SMALL_RESPONSE)


        # Создаем модель для генерации эмбеддингов
        self.yandex_embeddings = YandexGPTEmbeddings(
            model="doc",
            folder_id=YC_FOLDER_ID,
            iam_token=YC_API_KEY
        )

        # Инициализация индекса (только загрузка)
        self.faiss_index = self._load_faiss_index()

        self.results = None

    def _load_faiss_index(self):
        """
        Загружает FAISS индекс из локального файла

        Returns:
            FAISS: Векторное хранилище
        """
        index_path = "faiss_index"

        if os.path.exists(index_path):
            print(f"Загрузка существующего индекса из {index_path}")
            return FAISS.load_local(
                index_path,
                self.yandex_embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            raise FileNotFoundError(
                f"Файл FAISS индекса не найден: {index_path}. "
                "Сначала необходимо создать индекс локально."
            )

    def run_evaluation(self,):
        """
        Запускает полную оценку RAG системы

        Returns:
            Результаты оценки
            формат
            {'faithfulness': 0.7321, 'context_precision': 0.9286, 'context_recall': 1.0000}
        """

        metrics = [Faithfulness(), ContextPrecision(), ContextRecall()]

        # Создаем модель для оценки
        small_llm = YandexGPT(
            folder_id=self.folder_id,
            iam_token=self.api_key,
            model="yandexgpt-lite",
            temperature=0.3,
            max_tokens=500
        )

        # Формируем результаты оценки
        results = {}
        for key, q in self.questions.items():
            results[key] = {
                "question": q["text"],
                "response": q["response"],
                "ground_truth": q.get("ground_truth", "")
            }

        # Получаем релевантные контексты из индекса
        for r in results.values():
            relevants = self.faiss_index.similarity_search(r["question"], k=2)
            r["retrieved_contexts"] = [doc.page_content for doc in relevants]

        # Создаем датасет для оценки
        df_eval = pd.DataFrame([
            {
                "question": r["question"],
                "retrieved_contexts": r["retrieved_contexts"],
                "answer": r["response"],
                "ground_truth": r["ground_truth"]
            }
            for r in results.values()
        ])

        hf_ds = Dataset.from_pandas(df_eval)

        # Вычисляем метрики
        result = evaluate(
            dataset=hf_ds,
            metrics=metrics,
            llm=small_llm,
            embeddings=self.yandex_embeddings,
            show_progress=True,
            raise_exceptions=True,
            batch_size=4,
        )

        try:
            details_rag = result.to_pandas()
        except AttributeError:
            details_rag = pd.DataFrame(result)

        wanted = [
            "faithfulness",
            "context_precision",
            "context_recall",
        ]

        # Собираем словарь только из нужных метрик
        summary = {c: float(details_rag[c].mean(skipna=True)) for c in wanted if c in details_rag.columns}

        return summary


# Инициализация оценщика
evaluator = RAGEvaluator(folder_id=YC_FOLDER_ID, api_key=YC_API_KEY)

# Запуск оценки
results = evaluator.run_evaluation()

print(results)