let LANG = 'en'; try { LANG = localStorage.getItem('agrosense.lang') || 'en'; } catch {}
const t = (k, v) => (LANG === 'hi' && window.HI[k] ? window.HI[k] : k).replace(/\{(\w+)\}/g, (_, x) => v && v[x] != null ? v[x] : '');
function applyLang() {
  document.documentElement.lang = LANG;
  document.querySelectorAll('[data-i]').forEach(e => { e.dataset.en = e.dataset.en || e.textContent.trim(); e.textContent = t(e.dataset.en); });
  document.querySelectorAll('[data-ip]').forEach(e => e.placeholder = t(e.dataset.ip));
  const b = document.getElementById('langBtn'); if (b) b.textContent = LANG === 'hi' ? 'English' : 'हिन्दी';
}
function setLang(l) { LANG = l; try { localStorage.setItem('agrosense.lang', l); } catch {} applyLang(); if (window.onLang) window.onLang(); }
document.getElementById('langBtn').onclick = () => setLang(LANG === 'hi' ? 'en' : 'hi');
