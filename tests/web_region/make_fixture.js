// tests/web_region/ の fixture を作る: 合成の騎士（make_knight() を 20 倍）を画面で変換し、範囲を描いて保存する。
// node knight_fixture.js <sprite-app.html> <knight.png> <出力フォルダ>
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
const [,, html, png, outDir] = process.argv;
fs.mkdirSync(outDir, { recursive: true });
const log = [];
async function waitMotion(page) {
  await page.waitForTimeout(500);
  await page.waitForFunction(() => !document.querySelector('#motion-status .spinner') && (/動きを作りました/.test(document.getElementById('motion-status').textContent) || !document.getElementById('motion-warn').hidden || !document.getElementById('motion-error').hidden), null, { timeout: 120000 });
  await page.waitForTimeout(300);
}
async function save(page, sel, name) {
  const [d] = await Promise.all([page.waitForEvent('download'), page.click(sel)]);
  await d.saveAs(path.join(outDir, name));
}
async function saveTable(page, style, name) {
  await page.click('#table-save');
  await page.check(`input[name="spec-height"][value="${style}"]`);
  await save(page, '#table-save-go', name);
}
async function drag(page, a, b) {
  await page.$eval('#region-canvas', (e) => e.scrollIntoView({ block: 'center' }));
  const xy = (c, r) => page.evaluate(([c, r]) => {
    const q = document.getElementById('region-canvas').getBoundingClientRect(), k = editor.k, h = outImage.height;
    return { x: q.left + RULER_L + (c - 1) * k + k / 2, y: q.top + PAD + (h - r) * k + k / 2 };
  }, [c, r]);
  const p = await xy(...a), q = await xy(...b);
  await page.mouse.move(p.x, p.y); await page.mouse.down();
  for (let i = 1; i <= 6; i++) await page.mouse.move(p.x + (q.x - p.x) * i / 6, p.y + (q.y - p.y) * i / 6);
  await page.mouse.up();
}
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1200, height: 1500 }, acceptDownloads: true })).newPage();
  page.on('pageerror', (e) => log.push('pageerror: ' + e.message));
  await page.goto('file://' + html);
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('準備ができました'), null, { timeout: 180000 });
  log.push((await page.textContent('body')).match(/ビルド [0-9a-f]+/)[0]);
  await page.setInputFiles('#file', png);
  await page.click('#convert');
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('変換できました'), null, { timeout: 120000 });
  // 1. 呼吸の表を開き、絵の上で頭の四角（左 5〜右 24・下 10〜上 22）を描いて、upper_y をその四角で動かす（うなずくの形に直す）
  await page.check('#motion-breathe'); await waitMotion(page);
  await page.click('#table-open'); await waitMotion(page);
  await drag(page, [5, 22], [24, 10]); await waitMotion(page);
  await page.click('#region-use button >> text=upper_y'); await waitMotion(page);
  await page.fill('input[aria-label="1 行目の ms"]', '600');
  await page.fill('input[aria-label="2 行目の ms"]', '200'); await waitMotion(page);
  await page.fill('#table-name', 'web_rect'); await waitMotion(page);
  log.push('rect: ' + JSON.stringify(await page.evaluate(() => buildSpec())) + ' / ' + await page.textContent('#motion-meta'));
  await save(page, '#dl-sheet', 'web_rect_sheet.png'); await save(page, '#dl-json', 'web_rect_sheet.json');
  await saveTable(page, 'number', 'web_rect.number.motion.json'); await saveTable(page, 'percent', 'web_rect.percent.motion.json');
  // 2. 帯を足して頭の帯（下 10〜上 22）にし、名前を head_band に変え、upper_y をその帯で動かす。体全体のずれ root_y も付ける
  await page.click('#region-add-band'); await waitMotion(page);
  await page.fill('[data-edge-input="band:bottom"]', '10'); await waitMotion(page);
  await page.fill('[data-edge-input="band:top"]', '22'); await waitMotion(page);
  await page.fill('#region-tbody tr[data-region-row="band"] input.region-name', 'head_band');
  await page.press('#region-tbody tr[data-region-row="band"] input.region-name', 'Enter'); await waitMotion(page);
  await page.selectOption('select[data-col-region="upper_y"]', 'head_band'); await waitMotion(page);
  await page.click('#table-add'); await waitMotion(page);
  await page.fill('input[aria-label="3 行目の root_y"]', '-1'); await waitMotion(page);
  await page.fill('#table-name', 'web_band'); await waitMotion(page);
  log.push('band: ' + JSON.stringify(await page.evaluate(() => buildSpec())) + ' / ' + await page.textContent('#motion-meta'));
  await save(page, '#dl-sheet', 'web_band_sheet.png'); await save(page, '#dl-json', 'web_band_sheet.json');
  await saveTable(page, 'number', 'web_band.number.motion.json'); await saveTable(page, 'percent', 'web_band.percent.motion.json');
  await browser.close();
  console.log(log.join('\n'));
})().catch((e) => { console.error(e); console.log(log.join('\n')); process.exit(1); });
