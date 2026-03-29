import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)  # разрешаем запросы с GitHub Pages

# Загружаем модель один раз при старте сервера
# "base" - баланс скорости и точности. Можно "small" для лучшей точности
print("Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model loaded!")


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Lecture Helper API is running"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    # Сохраняем аудио во временный файл
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Транскрибируем
        segments, info = model.transcribe(
            tmp_path,
            language="en",
            beam_size=5,
            vad_filter=True,           # фильтр тишины
            vad_parameters=dict(
                min_silence_duration_ms=500
            )
        )

        # Собираем текст из сегментов
        text = " ".join(segment.text.strip() for segment in segments)
        text = text.strip()

        return jsonify({"text": text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
