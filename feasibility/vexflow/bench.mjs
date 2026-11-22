// Headless Chrome benchmark: node bench.mjs [path to chrome]  ->  results.json, screenshots/
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const chrome = process.argv[2] || process.env.CHROME || '/usr/bin/google-chrome';
const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 900 });
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto('file://' + path.join(here, 'dist', 'index.html'));
await page.waitForFunction('window.ready === true', { timeout: 120000 });
fs.mkdirSync(path.join(here, 'screenshots'), { recursive: true });
await page.screenshot({ path: path.join(here, 'screenshots', 'page.png') });
await (await page.$('#glyphs')).screenshot({ path: path.join(here, 'screenshots', 'accidentals.png') });
const res = await page.evaluate('bench()');
res.errors = errors;
fs.writeFileSync(path.join(here, 'results.json'), JSON.stringify(res, null, 1));
for (const p of res.pieces) {
  console.log(`${p.symbols} symbols: whole piece ${p.whole_piece.to_screen_ms.toFixed(0)} ms (${p.whole_piece.dom_nodes} nodes), ` +
    `visible only ${p.visible_only.to_screen_ms.toFixed(0)} ms, system ${p.system_draw_ms.toFixed(1)} ms, ` +
    `edit ${p.edit.move.median_ms.toFixed(1)} ms, insert ${p.edit.insert.median_ms.toFixed(1)}/${p.edit.insert.max_ms.toFixed(1)} ms, click ${p.click_selects_note}`);
}
if (errors.length) console.log('errors:', errors.slice(0, 5));
await browser.close();
