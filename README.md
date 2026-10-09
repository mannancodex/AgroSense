# AgroSense
Heatwave detection, smart irrigation and crop-growth advisory for smallholder farmers (Flask + scikit-learn + PyTorch).

## Run
    pip install -r requirements.txt
    python train_models.py     # optional: trained models are already in ./models
    python app.py              # open http://localhost:5000

## Layout
- `train_models.py` trains RandomForest heat classifier, RandomForest irrigation regressor (Hargreaves ET0 x FAO-56 Kc), PyTorch growth-health net (7,169 params). Writes `models/metadata.json`, the feature-order contract used by `app.py`.
- `app.py` serves the UI, `POST /api/analyze` and `GET /api/health`. Stateless; 404 for unknown places, 502 for weather-service failures.
- `agrosense.db` (SQLite, created on first run) holds `farmers` (name, mobile, village, district, state, hashed password) and `fields` (crop, stage, area, town, lat/lon, latest readings). Each farmer only sees their own fields.
- Pages: `/register`, `/login`, then the console at `/`. Set `AGROSENSE_SECRET` in production and serve behind HTTPS (e.g. gunicorn).

## Language and address
- The हिन्दी / English button works on every page and remembers your choice. All translations are in `hi.json` (English text -> Hindi). To add or fix a translation, edit that file only.
- Registration fills State, District and Village from the PIN code (via api.postalpincode.in, called by the server). If the PIN is not found, choose "Other" and type the value.
