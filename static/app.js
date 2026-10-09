const $ = id => document.getElementById(id);
const HC = ['#5f9e4a', '#d9a82b', '#e0782f', '#b5322a'];
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let fields = [];

async function api(url, method = 'GET', body) {
  const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json', 'X-Lang': LANG }, body: body && JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (r.status === 401) { location = '/login'; throw new Error(d.error); }
  if (!r.ok) throw new Error(d.error || 'Something went wrong.');
  return d;
}
$('logout').onclick = async () => { await api('/api/logout', 'POST'); location = '/login'; };
$('sm').oninput = () => $('smOut').textContent = $('sm').value + '%';

function current() { return fields.find(f => String(f.id) === $('fieldSel').value); }
function renderFields(selectId) {
  const sel = $('fieldSel'); sel.innerHTML = '';
  if (!fields.length) { sel.innerHTML = `<option value="">${t('No fields yet. Register one below')}</option>`; $('addBox').open = true; }
  fields.forEach(f => { const o = new Option(`${f.name} | ${t(f.crop)} | ${f.town}`, f.id); sel.add(o); });
  if (selectId) sel.value = selectId;
  pick();
}
function pick() {
  const f = current(), info = $('fieldInfo');
  info.hidden = $('delField').hidden = !f; if (!f) return;
  info.innerHTML = `<dl><dt>${t('Crop')}</dt><dd>${esc(t(f.crop))}</dd><dt>${t('Area')}</dt><dd>${f.area} ${t('acres')}</dd><dt>${t('Location')}</dt><dd>${esc(f.place || f.town)}</dd>
    <dt>${t('Coordinates')}</dt><dd>${f.lat.toFixed(3)}, ${f.lon.toFixed(3)}</dd></dl>`;
  $('stage').value = f.stage; $('sm').value = f.sm; $('smOut').textContent = Math.round(f.sm) + '%'; $('n').value = f.n; $('p').value = f.p; $('k').value = f.k;
}
$('fieldSel').onchange = pick;
$('delField').onclick = async () => {
  const f = current(); if (!f || !confirm(t('Remove field "{name}"? This cannot be undone.', { name: f.name }))) return;
  await api('/api/fields/' + f.id, 'DELETE'); fields = fields.filter(x => x.id !== f.id); renderFields();
};
$('fieldForm').onsubmit = async e => {
  e.preventDefault(); $('fErr').textContent = '';
  try {
    const { field } = await api('/api/fields', 'POST', { name: $('fName').value, town: $('fTown').value, crop: $('fCrop').value, stage: $('fStage').value, area: $('fArea').value });
    fields.push(field); fields.sort((a, b) => a.name.localeCompare(b.name)); renderFields(field.id); $('addBox').open = false; $('fieldForm').reset();
  } catch (err) { $('fErr').textContent = err.message; }
};

function show(text, isErr) {
  $('out').hidden = true; $('msg').hidden = false; $('msg').className = 'empty' + (isErr ? ' error' : '');
  $('msg').innerHTML = `<h2>${t(isErr ? 'We could not finish the analysis' : 'Working on it')}</h2><p>${esc(text)}</p>`;
}
$('form').onsubmit = async e => {
  e.preventDefault(); const f = current(); if (!f) return show(t('Register and choose a field first.'), true);
  const btn = $('run'); btn.disabled = true; btn.textContent = t('Analysing…'); show(t('Fetching weather and running the models.'));
  try {
    const d = await api('/api/analyze', 'POST', { field_id: f.id, stage: $('stage').value, soil_moisture: $('sm').value, n: $('n').value, p: $('p').value, k: $('k').value });
    Object.assign(f, { stage: $('stage').value, sm: +$('sm').value, n: +$('n').value, p: +$('p').value, k: +$('k').value }); render(d);
  } catch (err) { show(err.message || t('Could not reach AgroSense. Check your connection.'), true); }
  btn.disabled = false; btn.textContent = t('Run field analysis');
};

function render(d) {
  $('msg').hidden = true; $('out').hidden = false;
  $('place').textContent = `${d.field} | ${t(d.crop)} | ${d.location}`;
  const w = d.weather;
  $('wx').innerHTML = [[Math.round(w.temp) + '°C', t('now')], [Math.round(w.tmax) + '° / ' + Math.round(w.tmin) + '°', t('high / low today')], [Math.round(w.humidity) + '%', t('humidity')], [Math.round(w.wind) + ' km/h', t('wind')]]
    .map(([a, b]) => `<div><b>${a}</b><span>${b}</span></div>`).join('');
  const h = d.heat; $('heatLabel').textContent = h.label; $('heatLabel').style.color = HC[h.level];
  $('thumb').style.left = `calc(${[0, 33, 66, 97][h.level]}% - 3px)`;
  $('heatNote').textContent = t("{d}°C from this crop's seasonal normal. Model confidence {c}%.", { d: (h.departure >= 0 ? '+' : '') + h.departure, c: h.confidence });
  const i = d.irrigation; $('irrMm').textContent = i.mm;
  $('irrAcre').textContent = t('{l} litres/acre/day', { l: i.l_per_acre.toLocaleString('en-IN') });
  $('irrField').textContent = t('Whole field ({area} acres): {l} litres', { area: d.area, l: Math.round(i.l_per_acre * d.area).toLocaleString('en-IN') });
  $('irrConf').textContent = t(i.needed ? 'Irrigate today.' : 'Skip irrigation today.') + ' ' + t('Confidence {c}%.', { c: i.confidence });
  const s = i.schedule; $('schedSum').textContent = s.summary;
  $('sessions').innerHTML = s.sessions.map(x => `<li><b>${t(x.label)}</b> ${t('{label} {start} to {end} ({day})', { label: '', start: x.start, end: x.end, day: t(x.day) }).trim()}<span>${t("{mm} mm, {l} litres/acre, {share}% of today's water", { mm: x.mm, l: x.l_per_acre.toLocaleString('en-IN'), share: x.share })}</span></li>`).join('');
  $('why').innerHTML = s.reasons.map(r => `<li>${esc(r)}</li>`).join('');
  $('nuts').innerHTML = d.nutrients.map(n => `<div class="nut"><div class="nl"><b>${n.name}</b><span class="st ${n.key}">${n.status}</span></div>
    <div class="bar"><i style="width:${Math.min(100, n.value)}%"></i><u style="left:${Math.min(100, n.target)}%"></u></div><small>${t('Your reading {v}. Target about {t}.', { v: n.value, t: n.target })}</small></div>`).join('');
  const g = d.growth, arc = $('arcFill'); $('arcNum').textContent = g.score; $('verdict').textContent = g.verdict;
  arc.style.stroke = g.score >= 80 ? '#3c6e47' : g.score >= 60 ? '#7ea63f' : g.score >= 40 ? '#e0782f' : '#b5322a';
  requestAnimationFrame(() => arc.style.strokeDashoffset = 220 * (1 - g.score / 100));
  $('recs').innerHTML = d.recommendations.map(t => `<li>${esc(t)}</li>`).join('');
  $('strip').innerHTML = d.forecast.map(f => `<div class="day" style="--c:${HC[f.level]}"><b>${new Date(f.date + 'T00:00').toLocaleDateString(LANG === 'hi' ? 'hi-IN' : 'en-IN', { weekday: 'short', day: 'numeric' })}</b>
    <span>${Math.round(f.tmax)}° / ${Math.round(f.tmin)}°</span><span>${t('{mm} mm rain', { mm: f.precip.toFixed(1) })}</span><span>${f.label}</span></div>`).join('');
}
api('/api/fields').then(d => { fields = d.fields; renderFields(); });

function hello() { $('hello').textContent = t('Namaste, {name}.', { name: $('hello').dataset.name }); }
window.onLang = () => { hello(); renderFields($('fieldSel').value); if (!$('out').hidden) $('form').requestSubmit(); };
hello(); applyLang();
