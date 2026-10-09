"""AgroSense - Flask backend with farmer accounts (SQLite), field registry and ML advisory."""
import json, math, os, re, secrets, sqlite3
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
import joblib, numpy as np, requests, torch
from flask import Flask, g, jsonify, redirect, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash
from model_def import GrowthNet
from functools import lru_cache

ROOT = Path(__file__).parent; MD = ROOT / "models"
DB = os.environ.get("AGROSENSE_DB", str(ROOT / "agrosense.db"))
app = Flask(__name__)
_kf = ROOT / ".secret_key"
if not _kf.exists(): _kf.write_text(secrets.token_hex(32))
app.config.update(SECRET_KEY=os.environ.get("AGROSENSE_SECRET", _kf.read_text()), SESSION_COOKIE_HTTPONLY=True,
                  SESSION_COOKIE_SAMESITE="Lax", PERMANENT_SESSION_LIFETIME=timedelta(days=14))
LEVELS = ["Normal", "Heat Watch", "Heat Warning", "Severe Heatwave"]
ACRE = 4046.86
M = {}
HI = json.loads((ROOT / "hi.json").read_text(encoding="utf-8"))
_TPL = [(re.compile("^" + re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>.+?)", re.escape(k)) + "$"), v) for k, v in HI.items() if "{" in k]
def L(): return "hi" if request.headers.get("X-Lang") == "hi" else "en"
def tr(s):
    """English template/text -> Hindi when the client asked for it (exact match, else template match)."""
    if L() != "hi" or not isinstance(s, str): return s
    if s in HI: return HI[s]
    for rx, v in _TPL:
        m = rx.match(s)
        if m: return v.format(**{k: HI.get(x, x) for k, x in m.groupdict().items()})
    return s

SCHEMA = """
CREATE TABLE IF NOT EXISTS farmers(id INTEGER PRIMARY KEY, name TEXT NOT NULL, mobile TEXT NOT NULL UNIQUE, village TEXT NOT NULL,
  district TEXT, state TEXT, pw_hash TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS fields(id INTEGER PRIMARY KEY, farmer_id INTEGER NOT NULL REFERENCES farmers(id) ON DELETE CASCADE,
  name TEXT NOT NULL, crop TEXT NOT NULL, stage TEXT NOT NULL, area REAL NOT NULL, town TEXT NOT NULL, place TEXT, lat REAL NOT NULL, lon REAL NOT NULL,
  sm REAL DEFAULT 40, n REAL DEFAULT 50, p REAL DEFAULT 45, k REAL DEFAULT 50, created TEXT DEFAULT CURRENT_TIMESTAMP);
"""
def init_db():
    c = sqlite3.connect(DB); c.executescript(SCHEMA); c.commit(); c.close()
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row; g.db.execute("PRAGMA foreign_keys=ON")
    return g.db
@app.teardown_appcontext
def _close(_):
    d = g.pop("db", None)
    if d: d.close()

def load_models():
    try:
        meta = json.loads((MD / "metadata.json").read_text())
        net = GrowthNet(len(meta["features"]["growth"]))
        net.load_state_dict(torch.load(MD / "growth_net.pt", map_location="cpu")); net.eval()
        M.update(meta=meta, clf=joblib.load(MD / "heat_clf.joblib"), reg=joblib.load(MD / "irrigation_reg.joblib"), net=net, ok=True)
    except Exception as e: M.update(ok=False, err=type(e).__name__)
init_db(); load_models()

class ApiError(Exception):
    def __init__(self, msg, code=400): self.msg, self.code = msg, code
@app.errorhandler(ApiError)
def _api_err(e): return jsonify(error=tr(e.msg)), e.code
@app.errorhandler(Exception)
def _any_err(e):
    code = getattr(e, "code", 500); code = code if isinstance(code, int) else 500
    return jsonify(error=tr("Something went wrong on our side. Please try again in a moment." if code == 500 else "That page or action was not found.")), code
def auth(f):
    @wraps(f)
    def w(*a, **k):
        if "uid" not in session: raise ApiError("Please log in to continue.", 401)
        return f(*a, **k)
    return w

def num(d, key, lo, hi, label):
    try: v = float(d[key])
    except (KeyError, TypeError, ValueError): raise ApiError(f"Please enter a number for {label}.")
    if not lo <= v <= hi: raise ApiError(f"{label} must be between {lo:g} and {hi:g}.")
    return v
def text(d, key, label, lo=1, hi=80, req=True):
    v = str(d.get(key) or "").strip()
    if (req and len(v) < lo) or len(v) > hi: raise ApiError(f"Please enter {label} ({lo}-{hi} characters)." if req else f"{label} is too long.")
    return v

# ---------- weather ----------
def geocode(q):
    try: r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": q, "count": 1}, timeout=10); r.raise_for_status()
    except requests.RequestException: raise ApiError("The weather service is not reachable right now. Please try again shortly.", 502)
    res = r.json().get("results")
    if not res: raise ApiError(f"We could not find a place called '{q}'. Check the spelling or add the district name.", 404)
    p = res[0]; return p["latitude"], p["longitude"], ", ".join(x for x in (p["name"], p.get("admin1"), p.get("country")) if x)

def forecast(lat, lon):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 7,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum,wind_speed_10m_max,relative_humidity_2m_mean,sunrise,sunset"})
        r.raise_for_status(); return r.json()
    except (requests.RequestException, ValueError): raise ApiError("The weather service is not reachable right now. Please try again shortly.", 502)

# ---------- models ----------
def nutrient_targets(ci, si):
    m = M["meta"]; o = list(m["stage_opt"][si]); o[0] = min(95, max(15, o[0] * m["n_factor"][ci])); return [round(x) for x in o]

def predict(w, ci, si, sm, n, p, k):
    meta, d = M["meta"], w["daily"]; cur = w.get("current", {}); rows = []
    for i, day in enumerate(d["time"]):
        doy = date.fromisoformat(day).timetuple().tm_yday
        normal = meta["crop_base"][ci] + 5 * math.sin(2 * math.pi * (doy - 60) / 365)
        hm = (d.get("relative_humidity_2m_mean") or [None] * 7)[i]
        tmax, tmin = d["temperature_2m_max"][i], d["temperature_2m_min"][i]
        rows.append(dict(date=day, tmax=tmax, tmin=tmin, humidity=hm if hm is not None else cur.get("relative_humidity_2m", 50),
                         wind=d["wind_speed_10m_max"][i] or 0, solar=d["shortwave_radiation_sum"][i] or 0, precip=d["precipitation_sum"][i] or 0,
                         soil_moisture=sm, doy=doy, crop=ci, stage=si, departure=tmax - normal))
    probs = M["clf"].predict_proba(np.array([[r[f] for f in meta["features"]["heat"]] for r in rows]))
    for r, pr in zip(rows, probs):
        full = np.zeros(4); full[M["clf"].classes_.astype(int)] = pr
        r.update(level=int(full.argmax()), confidence=round(float(full.max()) * 100), probs=[round(float(x), 3) for x in full])
    t = rows[0]
    per_tree = np.array([e.predict(np.array([[t[f] for f in meta["features"]["irrigation"]]]))[0] for e in M["reg"].estimators_])
    irr = max(0.0, float(per_tree.mean())); conf = round(100 / (1 + float(per_tree.std())))
    tg = nutrient_targets(ci, si)
    g_ = dict(t, n=n, p=p, k=k, n_gap=n - tg[0], p_gap=p - tg[1], k_gap=k - tg[2], heat_level=t["level"], irrigation=irr)
    xg = (np.array([[g_[f] for f in meta["features"]["growth"]]], dtype=np.float32) - meta["scaler"]["mean"]) / meta["scaler"]["std"]
    with torch.no_grad(): score = float(np.clip(M["net"](torch.tensor(xg, dtype=torch.float32)).item(), 0, 100))
    return rows, irr, conf, score

def verdict(s): return "Thriving" if s >= 80 else "Healthy" if s >= 60 else "Stressed" if s >= 40 else "Critical"

def nutrients(ci, si, n, p, k):
    out = []
    for name, v, t in zip(("Nitrogen", "Phosphorus", "Potassium"), (n, p, k), nutrient_targets(ci, si)):
        out.append(dict(name=name, value=round(v), target=t, status="Low" if v < t - 12 else "High" if v > t + 12 else "Balanced"))
    return out

# ---------- irrigation timing ----------
def clock(t): return f"{t.hour % 12 or 12}:{t.minute:02d} {'am' if t.hour < 12 else 'pm'}"

def schedule(w, irr, level):
    d, h = w["daily"], w["hourly"]; now = datetime.fromisoformat(w["current"]["time"])
    sr0, ss0 = datetime.fromisoformat(d["sunrise"][0]), datetime.fromisoformat(d["sunset"][0])
    sun = f"Sunrise is at {clock(sr0)} and sunset at {clock(ss0)} today."
    if irr < 0.5: return dict(sessions=[], summary="No irrigation needed today", reasons=["Soil moisture and today's rain already cover the crop's water demand.", sun])
    if (d["precipitation_sum"][0] or 0) >= 8:
        return dict(sessions=[], summary="Hold off, heavy rain is expected", reasons=[f"About {d['precipitation_sum'][0]:.0f} mm of rain is forecast today. Recheck tomorrow.", sun])
    times = [datetime.fromisoformat(t) for t in h["time"]]
    cost = [(h["temperature_2m"][i] or 0) + 0.4 * (h["wind_speed_10m"][i] or 0) + 0.3 * (h["precipitation_probability"][i] or 0) for i in range(len(times))]
    hour = timedelta(hours=1); cands = []
    for day in (0, 1):
        sr, ss = datetime.fromisoformat(d["sunrise"][day]), datetime.fromisoformat(d["sunset"][day])
        for name, a, b in (("Morning", sr, sr + timedelta(hours=3)), ("Evening", ss - timedelta(hours=2), ss)):
            a = max(a, now + timedelta(minutes=15))
            if b <= a: continue
            idx = [i for i, t in enumerate(times) if t + hour > a and t < b]
            if not idx: continue
            best = min(((i, i + 1) if i + 1 in idx else (i,) for i in idx), key=lambda g_: sum(cost[j] for j in g_) / len(g_))
            s, e = max(times[best[0]], a), min(times[best[-1]] + hour, b)
            if (e - s) < timedelta(minutes=30): s = max(a, e - timedelta(minutes=45))
            cands.append(dict(label=name, day=day, start=s, end=e, cost=sum(cost[j] for j in best) / len(best),
                              temp=h["temperature_2m"][best[0]], wind=max(h["wind_speed_10m"][j] or 0 for j in best), rain=max(h["precipitation_probability"][j] or 0 for j in best)))
    today = [c for c in cands if c["day"] == 0] or cands[:1]
    split = (irr > 6 or level >= 2) and len(today) == 2
    chosen = today if split else [min(today, key=lambda c: c["cost"])]
    shares = {"Morning": .6, "Evening": .4} if split else {chosen[0]["label"]: 1}
    sessions = [dict(label=c["label"], day="Today" if c["day"] == 0 else "Tomorrow", start=clock(c["start"]), end=clock(c["end"]), share=round(shares[c["label"]] * 100),
                     mm=round(irr * shares[c["label"]], 1), l_per_acre=round(irr * shares[c["label"]] * ACRE)) for c in chosen]
    mid = [i for i, t in enumerate(times[:24]) if 10 <= t.hour < 16]
    peak = max(h["temperature_2m"][i] or 0 for i in mid) if mid else 0
    why = [sun]
    if peak >= 32: why.append(f"Skipping 10 am to 4 pm: it reaches about {peak:.0f}°C, so much of the water would evaporate.")
    if level >= 2: why.append("Heat risk is high, so the water is " + ("split across the cool morning and evening." if split else "given in the coolest hours."))
    elif split: why.append("Water need is above 6 mm, so it is split in two lighter sessions to soak in rather than run off.")
    wmid = max((h["wind_speed_10m"][i] or 0 for i in mid), default=0)
    if wmid >= 20: why.append(f"Afternoon winds reach {wmid:.0f} km/h, which would blow sprinkler water off target.")
    if any(c["rain"] >= 60 for c in chosen): why.append("Rain is likely in this window. If it rains, skip the session.")
    if chosen[0]["day"] == 1: why.append("Today's cool windows have passed, so the next one is tomorrow morning.")
    summ = " and ".join(f"{s['label'].lower()} {s['start']} to {s['end']}" for s in sessions)
    return dict(sessions=sessions, summary="", reasons=why)

def recommendations(level, irr, verd, nut, stage):
    r = []
    if level >= 2: r.append("Cover soil with mulch or shade netting, and avoid watering at midday.")
    if level == 3: r.append("Delay transplanting and fertilizer application until the heatwave passes.")
    if irr > 6: r.append("Water need is high. Check pipes, drip lines and channels for leaks or blockages.")
    for x in nut:
        if x["status"] == "Low": r.append(f"{x['name']} is low for the {stage} stage (aim for about {x['target']}). Apply in small split doses and confirm the amount with your KVK.")
        elif x["status"] == "High": r.append(f"{x['name']} is higher than needed (aim for about {x['target']}). Skip extra {x['name'].lower()} to avoid root burn and wasted money.")
    if verd in ("Stressed", "Critical") and not any(x["status"] != "Balanced" for x in nut): r.append("Keep irrigation regular and look for wilting or leaf scorch.")
    return (r or ["Conditions are favourable. Continue your normal routine and recheck tomorrow."])[:5]

# ---------- pages & auth ----------
def farmer_row(): return db().execute("SELECT id,name,mobile,village,district,state FROM farmers WHERE id=?", (session["uid"],)).fetchone()
@app.get("/")
def index():
    if "uid" not in session: return redirect("/login")
    f = farmer_row()
    if not f: session.clear(); return redirect("/login")
    return render_template("index.html", hi=HI, farmer=dict(f), crops=M.get("meta", {}).get("crops", []), stages=M.get("meta", {}).get("stages", []))
@app.get("/login")
def login_page(): return redirect("/") if "uid" in session else render_template("auth.html", mode="login", hi=HI)
@app.get("/register")
def register_page(): return redirect("/") if "uid" in session else render_template("auth.html", mode="register", hi=HI)

@app.post("/api/register")
def register():
    d = request.get_json(silent=True) or {}
    name, village = text(d, "name", "your name", 2, 60), text(d, "village", "your village or town", 2, 60)
    mobile = re.sub(r"[\s-]", "", str(d.get("mobile", ""))).removeprefix("+91")
    if not re.fullmatch(r"[6-9]\d{9}", mobile): raise ApiError("Enter a valid 10-digit Indian mobile number.")
    pw = str(d.get("password", ""))
    if len(pw) < 8: raise ApiError("Choose a password with at least 8 characters.")
    try:
        cur = db().execute("INSERT INTO farmers(name,mobile,village,district,state,pw_hash) VALUES(?,?,?,?,?,?)",
                           (name, mobile, village, text(d, "district", "your district", 2, 60), text(d, "state", "your state", 2, 60), generate_password_hash(pw)))
        db().commit()
    except sqlite3.IntegrityError: raise ApiError("This mobile number is already registered. Please log in instead.", 409)
    session.clear(); session.permanent = True; session["uid"] = cur.lastrowid; return jsonify(ok=True)

@app.post("/api/login")
def login():
    d = request.get_json(silent=True) or {}
    mobile = re.sub(r"[\s-]", "", str(d.get("mobile", ""))).removeprefix("+91")
    row = db().execute("SELECT id,pw_hash FROM farmers WHERE mobile=?", (mobile,)).fetchone()
    if not row or not check_password_hash(row["pw_hash"], str(d.get("password", ""))): raise ApiError("Mobile number or password is not correct.", 401)
    session.clear(); session.permanent = True; session["uid"] = row["id"]; return jsonify(ok=True)

@app.post("/api/logout")
def logout(): session.clear(); return jsonify(ok=True)

@app.get("/api/health")
def health(): return jsonify(status="ok" if M["ok"] else "models_not_loaded", **({"metrics": M["meta"]["metrics"]} if M["ok"] else {}))

@lru_cache(maxsize=256)
def _pin(pin):
    r = requests.get(f"https://api.postalpincode.in/pincode/{pin}", timeout=10); r.raise_for_status(); return r.json()
@app.get("/api/pincode/<pin>")
def pincode(pin):
    if not re.fullmatch(r"\d{6}", pin): raise ApiError("Enter a valid 6-digit PIN code.")
    try: data = _pin(pin)
    except (requests.RequestException, ValueError): raise ApiError("The PIN code service is not reachable right now. Please try again shortly.", 502)
    po = (data[0].get("PostOffice") if data else None)
    if not po: raise ApiError("We could not find this PIN code.", 404)
    return jsonify(state=po[0]["State"], districts=sorted({p["District"] for p in po}), villages=sorted({p["Name"] for p in po}))

# ---------- fields ----------
def field_dict(r): return {k: r[k] for k in r.keys() if k != "farmer_id"}
@app.get("/api/fields")
@auth
def list_fields(): return jsonify(fields=[field_dict(r) for r in db().execute("SELECT * FROM fields WHERE farmer_id=? ORDER BY name", (session["uid"],))])

@app.post("/api/fields")
@auth
def add_field():
    d = request.get_json(silent=True) or {}; meta = M["meta"]
    name, town = text(d, "name", "a field name", 1, 40), text(d, "town", "the town or village", 2, 80)
    if d.get("crop") not in meta["crops"]: raise ApiError("Please choose a crop from the list.")
    if d.get("stage") not in meta["stages"]: raise ApiError("Please choose a growth stage from the list.")
    area = num(d, "area", 0.1, 1000, "Field area")
    if db().execute("SELECT COUNT(*) FROM fields WHERE farmer_id=?", (session["uid"],)).fetchone()[0] >= 20: raise ApiError("You can register up to 20 fields.")
    if db().execute("SELECT 1 FROM fields WHERE farmer_id=? AND name=?", (session["uid"], name)).fetchone(): raise ApiError("You already have a field with this name.", 409)
    lat, lon, place = geocode(town)
    cur = db().execute("INSERT INTO fields(farmer_id,name,crop,stage,area,town,place,lat,lon) VALUES(?,?,?,?,?,?,?,?,?)", (session["uid"], name, d["crop"], d["stage"], area, town, place, lat, lon))
    db().commit(); return jsonify(field=field_dict(db().execute("SELECT * FROM fields WHERE id=?", (cur.lastrowid,)).fetchone())), 201

@app.delete("/api/fields/<int:fid>")
@auth
def del_field(fid):
    db().execute("DELETE FROM fields WHERE id=? AND farmer_id=?", (fid, session["uid"])); db().commit(); return jsonify(ok=True)

# ---------- analysis ----------
@app.post("/api/analyze")
@auth
def analyze():
    if not M["ok"]: raise ApiError("The advisory models are not ready. Ask the administrator to run train_models.py.", 503)
    d = request.get_json(silent=True) or {}; meta = M["meta"]
    try: fid = int(d.get("field_id"))
    except (TypeError, ValueError): raise ApiError("Choose one of your registered fields first.")
    f = db().execute("SELECT * FROM fields WHERE id=? AND farmer_id=?", (fid, session["uid"])).fetchone()
    if not f: raise ApiError("That field was not found.", 404)
    if d.get("stage") not in meta["stages"]: raise ApiError("Please choose a growth stage from the list.")
    sm = num(d, "soil_moisture", 0, 100, "Soil moisture")
    n, p, k = (num(d, key, 0, 200, label) for key, label in (("n", "Nitrogen"), ("p", "Phosphorus"), ("k", "Potassium")))
    w = forecast(f["lat"], f["lon"])
    ci, si = meta["crops"].index(f["crop"]), meta["stages"].index(d["stage"])
    rows, irr, conf, score = predict(w, ci, si, sm, n, p, k)
    db().execute("UPDATE fields SET stage=?,sm=?,n=?,p=?,k=? WHERE id=?", (d["stage"], sm, n, p, k, fid)); db().commit()
    t, verd, cur = rows[0], verdict(score), w.get("current", {})
    nut, sch = nutrients(ci, si, n, p, k), schedule(w, irr, t["level"])
    sch["reasons"] = [tr(x) for x in sch["reasons"]]; sch["summary"] = tr(sch["summary"])
    return jsonify(
        location=f["place"] or f["town"], field=f["name"], crop=f["crop"], area=f["area"],
        weather=dict(temp=cur.get("temperature_2m", t["tmax"]), humidity=cur.get("relative_humidity_2m", t["humidity"]), wind=cur.get("wind_speed_10m", t["wind"]), tmax=t["tmax"], tmin=t["tmin"], precip=t["precip"]),
        heat=dict(level=t["level"], label=tr(LEVELS[t["level"]]), confidence=t["confidence"], probs=t["probs"], departure=round(t["departure"], 1)),
        irrigation=dict(needed=irr >= 0.5, mm=round(irr, 1), l_per_acre=round(irr * ACRE), confidence=conf, schedule=sch),
        growth=dict(score=round(score), verdict=tr(verd)), nutrients=[dict(x, key=x["status"].lower(), name=tr(x["name"]), status=tr(x["status"])) for x in nut],
        recommendations=[tr(r) for r in recommendations(t["level"], irr, verd, nut, d["stage"])],
        forecast=[dict(date=r["date"], tmax=r["tmax"], tmin=r["tmin"], precip=r["precip"], level=r["level"], label=tr(LEVELS[r["level"]])) for r in rows])

if __name__ == "__main__": app.run(debug=False, host="0.0.0.0", port=5000)
