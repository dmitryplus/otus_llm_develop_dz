# utils.py
import os
import json
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def load_base_documents(filename: str, md_dir: str = "tasks") -> list:
    """
    Загружает базовые документы из JSON файла и обогащает их контекстом из .md файлов

    Args:
        filename: Путь к файлу
        md_dir: Директория с .md файлами

    Returns:
        Список обработанных документов
    """
    if os.path.exists(filename):
        print(f"Загрузка базовых документов из {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            documents = json.load(f)

        # Обогащаем документы контекстом из .md файлов
        for doc in documents:
            file_name = f"{doc['id']}.md"
            file_path = os.path.join(md_dir, file_name)

            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    doc["context"] = f.read()
            else:
                doc["context"] = ""

            doc["search"] = f"{doc['title']} {doc['context']}"

        return documents
    else:
        raise FileNotFoundError(f"Файл {filename} не найден")


def load_base_questions(filename: str) -> dict:
    """
    Загружает базовые вопросы из JSON файла

    Args:
        filename: Путь к файлу

    Returns:
        Словарь с вопросами
    """
    if os.path.exists(filename):
        print(f"Загрузка базовых вопросов из {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Файл {filename} не найден")


def save_questions_to_file(
    filename: str,
    field_name: str,
    llm,
    questions: dict
) -> dict:
    """
    Сохраняет/обновляет вопросы в файл с генерацией указанного поля

    Args:
        filename: Имя файла для сохранения
        field_name: Имя поля для генерации (например, "ground_truth")
        llm: Модель LLM для генерации значений
        questions: Исходный словарь вопросов

    Returns:
        Обновленный словарь вопросов
    """
    if os.path.exists(filename):
        print(f"Загрузка вопросов из {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        has_field = any(field_name in q for q in data.values())

        if not has_field:
            print(f"Генерация {field_name} для вопросов...")
            for key, q in data.items():
                q[field_name] = answer_question(llm, q["text"])

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        return data

    else:
        print(f"Сохранение новых вопросов в {filename}")
        for key, q in questions.items():
            q[field_name] = answer_question(llm, q["text"])

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        return questions


def answer_question(llm, query, faiss_index):
    """
    Отвечает на вопрос с использованием RAG подхода

    Args:
        llm: Модель LLM (например YandexGPT)
        query: Входной вопрос
        faiss_index: Векторный индекс FAISS

    Returns:
        Текст ответа модели
    """
    # Шаблон промпта
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""Вы — фактологичный помощник. Отвечайте кратко и точно.
Используйте следующие документы:

Контекст:
{context}

Вопрос:
{question}

Ответ:"""
    )

    # Создаем цепочку
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Получаем релевантные документы
    relevants = faiss_index.similarity_search(query, k=2)

    # Формируем контекст
    context = "\n\n".join([doc.page_content for doc in relevants])

    # Выполняем запрос
    result = llm_chain.invoke({
        "question": query,
        "context": context
    })

    return result["text"]