import os
import json
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Инициализация пути к бинарному артефакту модели (относительно скрипта)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_rf.pkl')

# Попытка загрузки сериализованного графа вычислений (Pipeline)
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print(f"Системное предупреждение: конфигурационный артефакт {MODEL_PATH} не удалось загрузить. Детали: {e}")

@app.route('/', methods=['GET'])
def index():
    # Инициализация базового вектора признаков для нагрузочного тестирования
    default_payload = json.dumps({
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
    }, indent=4)
    return render_template('index.html', default_payload=default_payload, model_status=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    # Проверка готовности вычислительного ядра
    if not model_loaded:
        return jsonify({"error": "Внутренняя ошибка сервера: Артефакт модели недоступен для чтения."}), 500

    try:
        # Извлечение JSON-массива из тела HTTP-запроса
        if request.is_json:
            data = request.get_json()
        else:
            # Альтернативный метод парсинга для запросов, переданных через веб-форму
            raw_text = request.form.get('json_data', '')
            if not raw_text:
                try:
                    data = json.loads(request.data)
                except:
                    return jsonify({"error": "Пустое тело запроса или неверный заголовок Content-Type."}), 400
            else:
                data = json.loads(raw_text)

        # Конвертация вектора признаков в структуру Pandas DataFrame для прохождения через Pipeline
        df = pd.DataFrame([data])
        
        # Получение распределения вероятностей и финального классификационного решения
        prob = model.predict_proba(df)[0][1]
        pred_class = model.predict(df)[0]
        
        return jsonify({
            "status": "success",
            "probability": round(prob, 4),
            "predicted_class": int(pred_class)
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Синтаксическая ошибка: переданный вектор не является валидным JSON-объектом."}), 400
    except Exception as e:
        # Перехват неконтролируемых исключений процессинга
        return jsonify({"error": f"Глобальное исключение в модуле предсказания: {str(e)}"}), 500

if __name__ == '__main__':
    # Активация слушателя на локальном интерфейсе
    app.run(host='0.0.0.0', port=5000, debug=True)
