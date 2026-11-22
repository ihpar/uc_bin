/* VexFlow feasibility test for the makam score editor.
 *
 * Questions: (1) can VexFlow draw the AEU comma accidentals, on notes and in key
 * signatures, and the usul time signatures; (2) is drawing fast enough for long
 * pieces; (3) can notes be selected by click and edited with a local redraw.
 *
 * Everything outside the drawing of one measure is our own code: accidental
 * state, line breaking, one SVG per system, drawing only the visible systems,
 * hit testing and editing. That is the part the editor has to own anyway.
 */
const VF = VexFlow;
const { Renderer, Stave, StaveNote, Voice, Formatter, Accidental, Beam, Dot, StaveTie, Tuplet,
        KeySignature, Fraction, Barline } = VF;

// AEU accidentals as SMuFL glyphs (all eight are in the Bravura font shipped with VexFlow)
const ACC = { b1: '', b4: '', b5: '', b8: '',
              '#1': '', '#4': '', '#5': '', '#8': '', n: 'n' };
const ACC_NAME = { b1: 'koma flat', b4: 'bakiye flat', b5: 'küçük mücenneb flat', b8: 'büyük mücenneb flat',
                   '#1': 'koma sharp', '#4': 'bakiye sharp', '#5': 'küçük mücenneb sharp', '#8': 'büyük mücenneb sharp' };
// key signatures of the makams used in the test (accidental, written pitch)
const KEYSIG = {
  rast: [['b1', 'b/4'], ['#4', 'f/5']],
  mahur: [['#4', 'f/5']],
  segah: [['b1', 'b/4'], ['b1', 'e/5'], ['#4', 'f/5']],
  hicaz: [['b4', 'b/4'], ['#4', 'f/5'], ['#4', 'c/5']],
  muhayyerkurdi: [['b5', 'b/4']],
  ussak: [['b1', 'b/4']],
};
const STEPS = ['c', 'd', 'e', 'f', 'g', 'a', 'b'];
const LINE_W = 1180, SYS_H = 150, PAD = 10;

const state = { piece: null, systems: [], widths: [], drawn: new Map(), virtual: true, sel: null, host: null };

// ---------------------------------------------------------------- one measure
function shownAccidentals(piece, mi) {
  /* An accidental is drawn when it differs from what the key signature and the
     earlier notes of the measure imply. */
  const def = {};
  for (const [a, p] of KEYSIG[piece.makam] || []) def[p[0]] = a;
  const cur = {}, out = [];
  for (const n of piece.measures[mi].n) {
    if (n.rest || n.tie) { out.push(null); continue; }
    const want = n.acc || null;
    const have = n.k in cur ? cur[n.k] : (def[n.k[0]] || null);
    out.push(want !== have ? (want || 'n') : null);
    cur[n.k] = want;
  }
  return out;
}

function buildMeasure(piece, mi) {
  const m = piece.measures[mi], show = shownAccidentals(piece, mi);
  const notes = m.n.map((n, i) => {
    const sn = new StaveNote({ keys: [n.rest ? 'b/4' : n.k], duration: n.d + (n.rest ? 'r' : ''), autoStem: true });
    for (let d = 0; d < (n.dots || 0); d++) Dot.buildAndAttach([sn], { all: true });
    if (show[i]) sn.addModifier(new Accidental(ACC[show[i]]), 0);
    return sn;
  });
  const voice = new Voice({ numBeats: piece.time[0], beatValue: piece.time[1] }).setMode(Voice.Mode.SOFT);
  voice.addTickables(notes);
  const tuplets = [];
  for (let i = 0; i + 2 < m.n.length; i++) {
    if (m.n[i].tup && m.n[i + 1].tup && m.n[i + 2].tup) { tuplets.push(new Tuplet(notes.slice(i, i + 3))); i += 2; }
  }
  const beams = Beam.generateBeams(notes, { groups: [new Fraction(1, 4)] });
  return { notes, voice, beams, tuplets };
}

function measureWidth(piece, mi) {
  const { voice } = buildMeasure(piece, mi);
  return new Formatter().joinVoices([voice]).preCalculateMinTotalWidth([voice]) * 1.25 + 28;
}

// ---------------------------------------------------------------- layout
function keySignature(piece) {
  const ks = new KeySignature('C');
  const sig = KEYSIG[piece.makam];
  if (sig && sig.length) ks.setOrdered(sig.map(([a, p]) => ({ type: ACC[a], pitch: p })));
  return ks;
}
function headWidth(piece, first) {
  return 46 + 14 * (KEYSIG[piece.makam] || []).length + (first ? 34 : 0);
}

function breakLines(piece, widths, fromMeasure = 0, systems = []) {
  /* Greedy line breaking. Systems before `fromMeasure` are kept, so an edit only
     re-flows the rest of the piece. */
  let keep = systems.findIndex(s => s.to > fromMeasure);
  if (keep < 0) keep = systems.length;
  const out = systems.slice(0, keep);
  let i = out.length ? out[out.length - 1].to : 0;
  while (i < widths.length) {
    const head = headWidth(piece, out.length === 0);
    let w = 0, j = i;
    while (j < widths.length && (j === i || w + widths[j] <= LINE_W - head)) w += widths[j++];
    out.push({ from: i, to: j, sum: w, head });
    i = j;
  }
  return out;
}

// ---------------------------------------------------------------- drawing
function drawSystem(si) {
  const { piece, systems, widths, host } = state, sys = systems[si];
  const div = document.createElement('div');
  div.className = 'sys'; div.style.top = si * SYS_H + 'px'; div.dataset.s = si;
  const r = new Renderer(div, Renderer.Backends.SVG);
  r.resize(LINE_W + 2 * PAD, SYS_H);
  const ctx = r.getContext();
  const last = sys.to === piece.measures.length;
  const scale = last && sys.sum < 0.6 * (LINE_W - sys.head) ? 1 : (LINE_W - sys.head) / sys.sum;
  let x = PAD, prev = null;
  for (let mi = sys.from; mi < sys.to; mi++) {
    const firstInSys = mi === sys.from;
    const w = widths[mi] * scale + (firstInSys ? sys.head : 0);
    const stave = new Stave(x, 32, w);
    if (firstInSys) {
      stave.addClef('treble').addModifier(keySignature(piece));
      if (si === 0) stave.addTimeSignature(piece.time[0] + '/' + piece.time[1]);
    }
    if (piece.measures[mi].c && !firstInSys) stave.setBegBarType(Barline.type.NONE);
    if (mi + 1 < piece.measures.length && piece.measures[mi + 1].c) stave.setEndBarType(Barline.type.DOTTED);
    if (mi + 1 === piece.measures.length) stave.setEndBarType(Barline.type.END);
    stave.setContext(ctx).draw();
    const b = buildMeasure(piece, mi);
    new Formatter().joinVoices([b.voice]).formatToStave([b.voice], stave);
    b.voice.draw(ctx, stave);
    b.beams.forEach(k => k.setContext(ctx).draw());
    b.tuplets.forEach(k => k.setContext(ctx).draw());
    b.notes.forEach((sn, i) => {
      const n = piece.measures[mi].n[i];
      if (n.tie) new StaveTie({ firstNote: prev, lastNote: sn, firstIndexes: [0], lastIndexes: [0] }).setContext(ctx).draw();
      if (!n.rest) prev = sn;
      const el = div.querySelector('#' + CSS.escape('vf-' + sn.getAttribute('id')));
      if (el) { el.dataset.m = mi; el.dataset.i = i; el.classList.add('hit');
                if (state.sel && state.sel.m === mi && state.sel.i === i) el.classList.add('sel'); }
    });
    x += w;
  }
  return div;
}

function visibleRange() {
  if (!state.virtual) return [0, state.systems.length];
  const top = state.host.parentElement.scrollTop, h = state.host.parentElement.clientHeight;
  return [Math.max(0, Math.floor(top / SYS_H) - 1), Math.min(state.systems.length, Math.ceil((top + h) / SYS_H) + 1)];
}

function refresh(force = []) {
  /* Draw the systems in view, drop the others. `force` lists systems to redraw. */
  const [a, b] = visibleRange();
  for (const [si, div] of state.drawn) {
    if (si < a || si >= b || force.includes(si) || force === 'all') { div.remove(); state.drawn.delete(si); }
  }
  let n = 0;
  for (let si = a; si < b; si++) {
    if (!state.drawn.has(si)) { const d = drawSystem(si); state.host.appendChild(d); state.drawn.set(si, d); n++; }
  }
  return n;
}

function load(piece, virtual = true) {
  const t0 = performance.now();
  state.piece = piece; state.virtual = virtual; state.sel = null;
  state.widths = piece.measures.map((_, i) => measureWidth(piece, i));
  const t1 = performance.now();
  state.systems = breakLines(piece, state.widths);
  const t2 = performance.now();
  state.host.innerHTML = ''; state.drawn.clear();
  state.host.style.height = state.systems.length * SYS_H + 'px';
  state.host.parentElement.scrollTop = 0;
  const n = refresh();
  const t3 = performance.now();
  return { widths_ms: t1 - t0, breaks_ms: t2 - t1, draw_ms: t3 - t2, systems_drawn: n, systems: state.systems.length };
}

// ---------------------------------------------------------------- editing
const systemOf = mi => state.systems.findIndex(s => mi >= s.from && mi < s.to);

function select(m, i) {
  state.host.querySelectorAll('.sel').forEach(e => e.classList.remove('sel'));
  state.sel = { m, i };
  const el = state.host.querySelector(`.hit[data-m="${m}"][data-i="${i}"]`);
  if (el) el.classList.add('sel');
}

function edit(fn) {
  /* Apply a change to the selected note; redraw one system, or re-flow from
     that system on when the measure width changed. Returns what was done. */
  const { m } = state.sel, t0 = performance.now();
  fn(state.piece.measures[m].n, state.sel.i);
  const w = measureWidth(state.piece, m), si = systemOf(m);
  let kind = 'one system', redrawn;
  if (Math.abs(w - state.widths[m]) > 0.5) {
    state.widths[m] = w;
    const before = state.systems;
    state.systems = breakLines(state.piece, state.widths, state.systems[si].from, state.systems);
    const changed = [];
    for (let k = si; k < state.systems.length; k++) {
      const o = before[k], s = state.systems[k];
      if (k > si && o && o.from === s.from && o.to === s.to) break;   // flow has settled
      changed.push(k);
    }
    state.host.style.height = state.systems.length * SYS_H + 'px';
    redrawn = refresh(changed); kind = `re-flow of ${changed.length} system(s)`;
  } else redrawn = refresh([si]);
  return { kind, redrawn, ms: performance.now() - t0 };
}

const ops = {
  step(d) { return edit((n, i) => { if (n[i].rest) return; const [l, o] = n[i].k.split('/');
    const p = STEPS.indexOf(l) + 7 * (+o) + d; n[i].k = STEPS[((p % 7) + 7) % 7] + '/' + Math.floor(p / 7); }); },
  accidental(a) { return edit((n, i) => { if (a) n[i].acc = a; else delete n[i].acc; }); },
  insert() { return edit((n, i) => { n.splice(i + 1, 0, { d: '16', k: n[i].k || 'b/4' }); }); },
  remove() { return edit((n, i) => { if (n.length > 1) { n.splice(i, 1); state.sel.i = Math.min(i, n.length - 1); } }); },
};

// ---------------------------------------------------------------- page
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const fmt = x => x.toFixed(x < 10 ? 1 : 0) + ' ms';

async function start() {
  await document.fonts.ready;
  state.host = document.getElementById('score');
  const sel = document.getElementById('piece'), status = document.getElementById('status');
  for (const p of PIECES) sel.add(new Option(`${p.makam} ${p.form}, ${p.usul} · ${p.notes.toLocaleString()} symbols`, p.name));
  const say = t => { status.textContent = t; };
  const open = async () => {
    const p = JSON.parse(JSON.stringify(PIECES.find(q => q.name === sel.value)));
    const r = load(p, document.getElementById('virtual').checked);
    await frame();
    say(`Loaded ${p.notes.toLocaleString()} symbols in ${p.measures.length} measures: widths ${fmt(r.widths_ms)}, ` +
        `line breaks ${fmt(r.breaks_ms)}, drawing ${r.systems_drawn} of ${r.systems} systems ${fmt(r.draw_ms)}.`);
  };
  sel.onchange = open; document.getElementById('virtual').onchange = open;
  state.host.parentElement.addEventListener('scroll', () => refresh(), { passive: true });
  state.host.addEventListener('click', e => {
    const el = e.target.closest('.hit');
    if (!el) return;
    select(+el.dataset.m, +el.dataset.i);
    const n = state.piece.measures[state.sel.m].n[state.sel.i];
    say(`Selected measure ${state.sel.m + 1}, note ${state.sel.i + 1}: ${n.rest ? 'rest' : n.k.replace('/', '')}` +
        (n.acc ? ' ' + ACC_NAME[n.acc] : '') + `. Arrows move it, 1–8 set a comma accidental, 0 clears it, I inserts, Delete removes.`);
  });
  document.addEventListener('keydown', e => {
    if (!state.sel) return;
    const accs = ['b1', 'b4', 'b5', 'b8', '#1', '#4', '#5', '#8'];
    let r = null;
    if (e.key === 'ArrowUp') r = ops.step(1);
    else if (e.key === 'ArrowDown') r = ops.step(-1);
    else if (e.key >= '1' && e.key <= '8') r = ops.accidental(accs[+e.key - 1]);
    else if (e.key === '0') r = ops.accidental(null);
    else if (e.key === 'i' || e.key === 'I') r = ops.insert();
    else if (e.key === 'Delete' || e.key === 'Backspace') r = ops.remove();
    else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      const d = e.key === 'ArrowRight' ? 1 : -1; let { m, i } = state.sel; i += d;
      if (i < 0 && m > 0) { m--; i = state.piece.measures[m].n.length - 1; }
      else if (i >= state.piece.measures[m].n.length && m + 1 < state.piece.measures.length) { m++; i = 0; }
      if (i >= 0 && i < state.piece.measures[m].n.length) select(m, i);
    } else return;
    e.preventDefault();
    if (r) say(`Edit redrawn in ${fmt(r.ms)} (${r.kind}).`);
  });
  drawGlyphTable();
  await open();
  window.ready = true;
}

function drawGlyphTable() {
  /* All eight AEU accidentals on notes, and a key signature built from them. */
  const div = document.getElementById('glyphs');
  const r = new Renderer(div, Renderer.Backends.SVG); r.resize(1000, 150);
  const ctx = r.getContext();
  const stave = new Stave(10, 20, 980).addClef('treble');
  stave.addModifier(new KeySignature('C').setOrdered([{ type: ACC.b1, pitch: 'b/4' }, { type: ACC.b4, pitch: 'e/5' }, { type: ACC['#4'], pitch: 'f/5' }]));
  stave.addTimeSignature('10/8').setContext(ctx).draw();
  const order = ['b1', 'b4', 'b5', 'b8', '#1', '#4', '#5', '#8'];
  const notes = order.map((a, i) => new StaveNote({ keys: [['b/4', 'e/5', 'a/4', 'd/5', 'f/4', 'c/5', 'g/4', 'f/5'][i]], duration: 'q', autoStem: true })
    .addModifier(new Accidental(ACC[a]), 0));
  const v = new Voice({ numBeats: 8, beatValue: 4 }).addTickables(notes);
  new Formatter().joinVoices([v]).formatToStave([v], stave);
  v.draw(ctx, stave);
  document.getElementById('glyphnames').textContent = 'Left to right: ' + order.map(a => ACC_NAME[a]).join(', ') + '.';
}

// ---------------------------------------------------------------- benchmark (used by bench.mjs)
async function bench() {
  const out = { userAgent: navigator.userAgent, pieces: [] };
  for (const src of PIECES) {
    const res = { name: src.name, symbols: src.notes, measures: src.measures.length };
    for (const virtual of [false, true]) {
      const runs = [];
      for (let k = 0; k < 3; k++) {
        const t0 = performance.now();
        const r = load(JSON.parse(JSON.stringify(src)), virtual);
        await frame();
        runs.push({ ...r, total_ms: r.widths_ms + r.breaks_ms + r.draw_ms, to_screen_ms: performance.now() - t0,
                    dom_nodes: state.host.querySelectorAll('*').length });
      }
      runs.sort((a, b) => a.to_screen_ms - b.to_screen_ms);
      res[virtual ? 'visible_only' : 'whole_piece'] = runs[1];           // median of three
    }
    // per-system cost, edits and scrolling, in visible-only mode
    const n = state.systems.length; let t = performance.now();
    const pick = Array.from({ length: Math.min(20, n) }, (_, k) => Math.floor(k * n / Math.min(20, n)));
    pick.forEach(si => drawSystem(si));
    res.system_draw_ms = (performance.now() - t) / pick.length;
    res.systems = n;
    const times = { move: [], accidental: [], insert: [], remove: [] };
    const visible = [...state.drawn.keys()];
    for (let k = 0; k < 30; k++) {
      const si = visible[k % visible.length], s = state.systems[si];
      const m = s.from + (k % (s.to - s.from)), i = k % state.piece.measures[m].n.length;
      select(m, i);
      times.move.push(ops.step(k % 2 ? 1 : -1).ms);
      times.accidental.push(ops.accidental(['b1', '#4', 'b5', null][k % 4]).ms);
      const a = ops.insert(); times.insert.push(a.ms);
      times.remove.push(ops.remove().ms);
    }
    const stat = a => { a = [...a].sort((x, y) => x - y); return { median_ms: a[a.length >> 1], max_ms: a[a.length - 1] }; };
    res.edit = Object.fromEntries(Object.entries(times).map(([k, v]) => [k, stat(v)]));
    // jump to 10 places in the piece: time until the newly visible systems are drawn
    const sc = state.host.parentElement, jumps = [];
    for (let k = 1; k <= 10; k++) {
      sc.scrollTop = (state.systems.length * SYS_H - sc.clientHeight) * k / 10;
      t = performance.now(); refresh(); jumps.push(performance.now() - t); await frame();
    }
    res.scroll_jump = stat(jumps);
    // hit testing: a real click on a note must select that note
    sc.scrollTop = 0; refresh(); await frame();
    const el = state.host.querySelector('.hit[data-i="1"]') || state.host.querySelector('.hit');
    const box = el.getBoundingClientRect();
    document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2)
      .dispatchEvent(new MouseEvent('click', { bubbles: true }));
    res.click_selects_note = !!state.sel && state.sel.m === +el.dataset.m && state.sel.i === +el.dataset.i;
    out.pieces.push(res);
  }
  return out;
}

window.addEventListener('load', start);
