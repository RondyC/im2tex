# 📝 Image-to-LaTeX: Распознавание математических формул с вниманием

## 📌 Описание проекта

Этот проект представляет собой нейросетевую модель **Seq2Seq с механизмом внимания**, которая преобразует изображения математических формул в **LaTeX-код**.  
Модель основана на **ResNet50** в качестве энкодера и **LSTM с механизмом внимания Bahdanau** в качестве декодера.  
В качестве обучающих данных использовался **датасет im2latex-100k**, загруженный с [Hugging Face](https://huggingface.co/datasets/yuntian-deng/im2latex-100k).

## 📂 Структура проекта

- `dataset.py` – Классы для загрузки и предобработки данных.
- `model.py` – Определение архитектуры модели (ResNet50 + Attention + LSTM).
- `train.py` – Код обучения модели.
- `inference.py` – Код инференса (предсказания LaTeX-кода по изображениям).
- `utils.py` – Вспомогательные функции.
- `README.md` – Этот файл.

## 📦 Установка и запуск

### 1️⃣ **Клонирование репозитория**
```bash
git clone https://github.com/your-repository/Image-to-LaTeX.git
cd Image-to-LaTeX

Используется im2latex-100k с Hugging Face, который уже содержит:

Обучающую выборку (55 033 примера)
Тестовую выборку (6 810 примеров)
Валидационную выборку (6 072 примера)

Архитектура модели
Модель состоит из:

ResNet50 (энкодер): Извлекает пространственные признаки из изображения.
LSTM (декодер): Генерирует последовательность токенов LaTeX.
Механизм внимания (Bahdanau): Фокусирует декодер на значимых участках изображения.
📈 Результаты обучения
Модель обучалась 3 эпохи, среднее время 39 минут на эпоху, скорость 1.36 s/it.
Потери на обучении:

Эпоха 1: 1.8487
Эпоха 2: 0.7084
Эпоха 3: 0.4705
