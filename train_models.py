"""Trains the three AgroSense models on synthetic, physically-grounded data.
Run:  python train_models.py     (writes ./models/*)"""
import json, joblib, numpy as np, torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from model_def import GrowthNet

rng = np.random.default_rng(42); torch.manual_seed(42)
CROPS = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Tomato", "Onion", "Soybean", "Groundnut", "Millet"]
STAGES = ["seedling", "vegetative", "flowering", "maturity"]
BASE = [27, 31, 30, 32, 31, 29, 28, 30, 31, 33]            # regional normal-temperature baseline per crop (C)
KC = np.array([[.4,1.15,1.15,.4],[1.05,1.2,1.2,.9],[.3,1.2,1.2,.6],[.35,1.15,1.15,.7],[.4,1.25,1.25,.75],
               [.6,1.15,1.15,.8],[.5,1.05,1.05,.75],[.4,1.15,1.15,.5],[.4,1.15,1.15,.6],[.3,1.0,1.0,.3]])  # FAO-56 style Kc
STAGE_OPT = [[40, 60, 40], [70, 45, 45], [55, 55, 70], [40, 35, 65]]   # target N,P,K index by growth stage
NF = [1, 1.15, 1.2, 1, 1.3, 1, .9, .5, .5, .8]                        # crop nitrogen-demand factor (legumes low)
N = 14000

def normal_temp(crop, doy):  # seasonal normal peaking around late May
    return np.array(BASE)[crop] + 5 * np.sin(2*np.pi*(doy-60)/365)

def heat_level(tmax, normal):  # IMD-style departure / absolute criteria
    dep = tmax - normal
    return np.select([(dep >= 6.5) | (tmax >= 45), (dep >= 4.5) | (tmax >= 43), (dep >= 3) | (tmax >= 40)], [3, 2, 1], 0)

def hargreaves_irrigation(tmax, tmin, rs, precip, sm, crop, stage):
    ra_mm = 0.408 * rs / 0.55                                # Rs ~ 0.55 Ra
    et0 = 0.0023 * ((tmax+tmin)/2 + 17.8) * np.sqrt(np.maximum(tmax-tmin, 0)) * ra_mm
    etc = et0 * KC[crop, stage]
    return np.maximum(0, etc * np.clip((80-sm)/60, 0, 1.2) - 0.5*precip)

doy = rng.integers(60, 300, N); crop = rng.integers(0, len(CROPS), N); stage = rng.integers(0, 4, N)
nrm = normal_temp(crop, doy)
tmax = nrm + rng.normal(2.5, 4, N); tmin = tmax - rng.uniform(8, 16, N)
hum = np.clip(rng.normal(50, 20, N), 10, 98); wind = rng.uniform(1, 25, N); rs = rng.uniform(8, 30, N)
precip = np.where(rng.random(N) < .2, rng.exponential(6, N), 0); sm = rng.uniform(3, 95, N)
npk = rng.uniform(5, 100, (N, 3))
W = dict(departure=tmax-nrm, tmax=tmax, tmin=tmin, humidity=hum, wind=wind, solar=rs, precip=precip, soil_moisture=sm,
         doy=doy, crop=crop, stage=stage)

# 1. Heatwave classifier
HF = ["tmax", "tmin", "humidity", "wind", "solar", "soil_moisture", "doy", "crop", "departure"]
level = heat_level(tmax, nrm)
X = np.column_stack([W[f] for f in HF]); Xtr, Xte, ytr, yte = train_test_split(X, level, test_size=.2, random_state=1)
clf = RandomForestClassifier(200, max_depth=18, random_state=1, n_jobs=-1).fit(Xtr, ytr)
acc = accuracy_score(yte, clf.predict(Xte))

# 2. Irrigation regressor
IF = ["tmax", "tmin", "solar", "humidity", "wind", "precip", "soil_moisture", "crop", "stage"]
irr = hargreaves_irrigation(tmax, tmin, rs, precip, sm, crop, stage)
X = np.column_stack([W[f] for f in IF]); Xtr, Xte, ytr, yte = train_test_split(X, irr, test_size=.2, random_state=1)
reg = RandomForestRegressor(120, min_samples_leaf=2, random_state=1, n_jobs=-1).fit(Xtr, ytr)
mae = mean_absolute_error(yte, np.maximum(reg.predict(Xte), 0))

# 3. Crop growth health (PyTorch)
GF = ["tmax", "tmin", "humidity", "wind", "solar", "soil_moisture", "crop", "stage", "n", "p", "k", "n_gap", "p_gap", "k_gap", "heat_level", "irrigation"]
opt = np.array(STAGE_OPT, float)[stage]; opt[:, 0] = np.clip(opt[:, 0]*np.array(NF)[crop], 15, 95); gap = npk - opt
pen = 12*level + np.maximum(np.abs(sm-55)-15, 0)*0.7 + (0.08*np.abs(gap) + 0.25*np.maximum(np.abs(gap)-10, 0)).sum(1) + np.maximum(irr-8, 0)*1.5 + np.maximum(tmin-28, 0)*1.0
score = np.clip(100 - 5 - pen + rng.normal(0, 3, N), 0, 100)
feats = dict(W, n=npk[:, 0], p=npk[:, 1], k=npk[:, 2], n_gap=gap[:, 0], p_gap=gap[:, 1], k_gap=gap[:, 2], heat_level=level, irrigation=irr)
X = np.column_stack([feats[f] for f in GF]).astype(np.float32)
mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X-mu)/sd
Xtr, Xte, ytr, yte = train_test_split(Xs, score.astype(np.float32), test_size=.2, random_state=1)
net = GrowthNet(len(GF)); opt = torch.optim.Adam(net.parameters(), 2e-3); lossf = nn.SmoothL1Loss()
Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
for ep in range(60):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 256):
        b = perm[i:i+256]; opt.zero_grad(); lossf(net(Xt[b]), yt[b]).backward(); opt.step()
with torch.no_grad(): nn_mae = float(torch.abs(net(torch.tensor(Xte)) - torch.tensor(yte)).mean())

joblib.dump(clf, "models/heat_clf.joblib", compress=3); joblib.dump(reg, "models/irrigation_reg.joblib", compress=3)
torch.save(net.state_dict(), "models/growth_net.pt")
json.dump(dict(crops=CROPS, crop_base=BASE, stage_opt=STAGE_OPT, n_factor=NF, stages=STAGES, features=dict(heat=HF, irrigation=IF, growth=GF),
               scaler=dict(mean=mu.tolist(), std=sd.tolist()),
               metrics=dict(heat_accuracy=round(acc, 4), irrigation_mae_mm=round(mae, 3), growth_mae_points=round(nn_mae, 2))),
          open("models/metadata.json", "w"), indent=2)
print(f"heat accuracy={acc:.3f}  irrigation MAE={mae:.2f} mm/day  growth MAE={nn_mae:.2f} pts  params={sum(p.numel() for p in net.parameters())}")
