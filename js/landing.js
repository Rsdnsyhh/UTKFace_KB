/* Navigasi halaman awal <-> aplikasi memakai hash (#app). Tidak mengubah logika di app.js. */
(function () {
  'use strict';
  const landing = document.getElementById('landing');
  const app = document.getElementById('app');

  function route() {
    const inApp = location.hash === '#app';
    landing.hidden = inApp;
    app.hidden = !inApp;
    if (inApp || location.hash !== '#tim') window.scrollTo(0, 0);
    document.title = inApp ? 'Estimasi Usia dari Wajah' : 'Estimasi Usia dari Wajah · Duo Hihu';
  }

  document.addEventListener('click', (e) => {
    const go = e.target.closest('[data-go]');
    if (go) { location.hash = go.dataset.go === 'app' ? 'app' : 'home'; }
    // tombol tema di halaman awal memakai tombol tema aplikasi
    if (e.target.closest('[data-theme-toggle]')) {
      const btn = document.getElementById('theme-toggle');
      if (btn) btn.click();
    }
  });
  window.addEventListener('hashchange', route);
  route();
})();
