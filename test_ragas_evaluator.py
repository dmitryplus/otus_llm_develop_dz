import os

import pytest
from ragas_evaluator import RAGEvaluator


YC_FOLDER_ID = os.getenv("folder_id")
YC_API_KEY = os.getenv("ya_token")


# Фикстура: подготовка RAGEvaluator
@pytest.fixture
def evaluator():
    folder_id = YC_FOLDER_ID
    api_key = YC_API_KEY

    # Инициализируем RAGEvaluator
    evaluator = RAGEvaluator(folder_id=folder_id, api_key=api_key)

    return evaluator


def test_run_evaluation(evaluator):
    result = evaluator.run_evaluation()

    # Проверка: Возвращается ли результат с нужными метриками
    assert "faithfulness" in result
    assert "context_precision" in result
    assert "context_recall" in result



def test_faithfulness_above_threshold(evaluator):
    result = evaluator.run_evaluation()

    # Проверка: Значение faithfulness больше 0.7
    assert result["faithfulness"] > 0.7, f"Faithfulness слишком низкий: {result['faithfulness']:.4f} <= 0.7"

    # Проверка: Значение context_precision больше 0.9
    assert result["context_precision"] > 0.9, f"context_precision слишком низкий: {result['context_precision']:.4f} <= 0.9"

    # Проверка: Значение context_recall больше 0.9
    assert result["context_recall"] > 0.9, f"context_recall слишком низкий: {result['context_recall']:.4f} <= 0.9"