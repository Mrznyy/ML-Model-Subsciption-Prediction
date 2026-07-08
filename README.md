# ML Predictor - REST API для классификации целевого действия

Flask-приложение для развертывания обученной модели машинного обучения (Random Forest) с веб-интерфейсом для тестирования. Сервис принимает признаки пользовательской сессии и возвращает вероятность целевого действия.

## Цель и ожидаемый результат

Компания «СберАвтоподписка» хочет увеличить эффективность сайта: улучшить пользовательский опыт, повысить конверсию, сделать рекламные кампании более результативными. Для этого создана модель, которая предсказывает вероятность того, что пользователь совершит целевое действие (оставит заявку, закажет звонок и пр.) на сайте.

Эта модель помогает:

- Оценивать эффективность каналов привлечения трафика;
- Адаптировать рекламные кампании;
- Улучшать UX сайта за счет анализа поведения пользователей.

## Структура репозитория

├── app.py # Flask-приложение  
├── model_rf.pkl # Сериализованный Pipeline модели  
├── model.ipynb # Jupyter Notebook: анализ, обучение и сравнение моделей  
├── requirements.txt # Зависимости Python  
├── README.md # Документация  
├── templates/  
│ └── index.html # Веб-интерфейс для тестирования  
├── data/  
│ ├── ga_hits.csv # Данные о действиях пользователей  
│ └── ga_sessions.csv # Данные о сессиях  


## Установка и запуск

### 1. Клонирование и окружение

```bash
git clone <repository-url>
cd <project-folder>
python -m venv venv
```

### 2. Активация виртуального окружения

Windows:
```bash
venv\Scripts\activate
```
macOS / Linux:
```bash
source venv/bin/activate
```

### 3. Установка зависимостей
Требуемые библиотеки:
```bash
pip install -r requirements.txt
```

### 4. Загрузка модели
Поместите файл model_rf.pkl в корневую директорию. Если файл отсутствует, API предсказаний будет недоступен (ошибка 500).

### 5. Запуск сервера
Запуск виртуального сервера командой:
```bash
python app.py
```
После запуска приложение становится доступным по адресу `http://127.0.0.1:5000/`. На главной странице размещена форма для ввода JSON-данных и тестирования модели в ручном режиме. 

## Использование API

`GET /` Возвращает HTML-страницу с веб-формой для ручного тестирования.

`POST /predict` Принимает JSON с признаками сессии, возвращает предсказание модели.

**Пример запроса:**

```json
{
    "visit_number": 1,
    "utm_medium": "cpm",
    "utm_campaign": "FTjNLDyTrXaWYgZymFkV",
    "utm_adcontent": "PkybGvWbaqORmxjNunqZ",
    "utm_keyword": "(not set)",
    "device_category": "mobile",
    "device_os": "Android",
    "device_brand": "Samsung",
    "device_browser": "Chrome",
    "geo_country": "Russia",
    "is_organic": 0,
    "is_social": 0,
    "visit_month": 9,
    "visit_dayofweek": 6,
    "visit_hour": 16,
    "city_top": "Gelendzhik",
    "source_top": "MvfHsxITijuriZxsqZqt"
}
```
**Пример ответа:**
```json
{
    "status": "success",
    "probability": 0.2782,
    "predicted_class": 0
}
```
<img width="721" height="686" alt="image" src="https://github.com/user-attachments/assets/379ed166-cb2d-4aea-aa73-6fff2c957595" />
