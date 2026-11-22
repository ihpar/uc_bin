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

/* Accidentals. A note carries a number of Holdrian commas, flat or sharp ("b2", "#4").
   AEU notation has special signs for 1, 4, 5 and 8 commas (SMuFL glyphs in Bravura);
   the numbered notation used by Mus2 writes a flat or sharp with the comma count.
   SymbTr gives every note in both spellings, and they can differ (AEU b1 = numbered b2). */
const AEU = { b1: '', b4: '', b5: '', b8: '',
              '#1': '', '#4': '', '#5': '', '#8': '' };
const AEU_NAME = { 1: 'koma', 4: 'bakiye', 5: 'küçük mücenneb', 8: 'büyük mücenneb' };
function glyph(acc) {
  if (acc === 'n') return 'n';
  if (state.accMode === 'aeu' && AEU[acc]) return AEU[acc];
  return (acc[0] === 'b' ? '' : '') + String.fromCodePoint(0xE880 + +acc.slice(1));
}
function accName(acc) {
  const n = +acc.slice(1), kind = acc[0] === 'b' ? 'flat' : 'sharp';
  return `${n}-comma ${kind}` + (AEU_NAME[n] ? ` (${AEU_NAME[n]})` : '');
}
// key signature (read from the mu2 file) and note accidental in the spelling that is on screen
const signature = p => (state.accMode === 'aeu' ? p.keysig : p.keysig53) || [];
const spelled = n => { const a = state.accMode === 'aeu' ? n.acc : (n.a53 || n.acc); return a && a !== 'n' ? a : null; };
const STEPS = ['c', 'd', 'e', 'f', 'g', 'a', 'b'];
const LINE_W = 1180, SYS_H = 150, PAD = 10;

const state = { accMode: 'aeu', piece: null, systems: [], widths: [], drawn: new Map(), virtual: true, sel: null, host: null };

// ---------------------------------------------------------------- one measure
function shownAccidentals(piece, mi) {
  /* An accidental is drawn when it differs from what the key signature and the
     earlier notes of the measure imply. */
  const def = {};
  for (const [a, p] of signature(piece)) def[p[0]] = a;
  const cur = {}, out = [];
  for (const n of piece.measures[mi].n) {
    if (n.rest || n.tie) { out.push(null); continue; }
    const want = spelled(n);
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
    if (show[i]) sn.addModifier(new Accidental(glyph(show[i])), 0);
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
  const sig = signature(piece);
  if (sig && sig.length) ks.setOrdered(sig.map(([a, p]) => ({ type: glyph(a), pitch: p })));
  return ks;
}
function headWidth(piece, first) {
  return 46 + (state.accMode === 'aeu' ? 14 : 24) * signature(piece).length + (first ? 34 : 0);
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
  accidental(a) { return edit((n, i) => { delete n[i].a53; if (a) n[i].acc = a; else delete n[i].acc; }); },
  insert() { return edit((n, i) => { n.splice(i + 1, 0, { d: '16', k: n[i].k || 'b/4' }); }); },
  remove() { return edit((n, i) => { if (n.length > 1) { n.splice(i, 1); state.sel.i = Math.min(i, n.length - 1); } }); },
};


// ---------------------------------------------------------------- playback
/* Pitch: the staff position and the comma accidental give the pitch in Holdrian
   commas (53 to the octave); frequency = 440 Hz * 2^(commas from A4 / 53).
   Sound: synthesized in the browser, so any comma pitch is exact. An instrument
   is one function that starts a note; sampled instruments would plug in the same way. */
const NATURAL = { c: 0, d: 9, e: 18, f: 22, g: 31, a: 40, b: 49 };
function frequency(n) {
  const [l, o] = n.k.split('/'), a = n.a53 || n.acc;           // the numbered spelling is the sounding pitch
  const shift = a && a !== 'n' ? (a[0] === 'b' ? -1 : 1) * +a.slice(1) : 0;
  return 440 * Math.pow(2, (NATURAL[l] + 53 * (+o - 4) + shift - 40) / 53);
}
function seconds(n, bpm) {
  let whole = 1 / +n.d;
  whole *= (2 ** ((n.dots || 0) + 1) - 1) / 2 ** (n.dots || 0);
  if (n.tup) whole *= 2 / 3;
  return whole * 4 * 60 / bpm;
}

const audio = { ctx: null, out: null, plucks: new Map() };
function audioContext() {
  if (!audio.ctx) {
    audio.ctx = new (window.AudioContext || window.webkitAudioContext)();
    audio.out = audio.ctx.createGain(); audio.out.gain.value = 0.5;
    const comp = audio.ctx.createDynamicsCompressor();
    audio.out.connect(comp).connect(audio.ctx.destination);
  }
  return audio.ctx;
}
function pluckBuffer(freq) {
  /* Plucked-string sound as a sum of decaying harmonics of exactly `freq`
     (higher harmonics are weaker and die away faster). Cached per pitch. */
  const ctx = audio.ctx, key = Math.round(freq * 100);
  if (audio.plucks.has(key)) return audio.plucks.get(key);
  const sr = ctx.sampleRate, len = Math.floor(sr * 2.5);
  const buf = ctx.createBuffer(1, len, sr), d = buf.getChannelData(0);
  const H = Math.min(14, Math.floor(0.45 * sr / freq));
  let norm = 0;
  for (let h = 1; h <= H; h++) {
    const amp = Math.sin(Math.PI * h * 0.23) / h, w = 2 * Math.PI * freq * h / sr;   // plucked at 23% of the string
    const decay = Math.exp(-(1.6 + 0.5 * h) / sr);
    let env = 1; norm += Math.abs(amp);
    for (let i = 0; i < len; i++) { d[i] += amp * env * Math.sin(w * i); env *= decay; }
  }
  const attack = Math.floor(sr * 0.002);
  for (let i = 0; i < len; i++) d[i] *= (i < attack ? i / attack : 1) / norm;
  const v = { buf, rate: 1 };
  audio.plucks.set(key, v);
  return v;
}
const INSTRUMENTS = {
  pluck: { label: 'Plucked string (synthesized)', start(freq, t, dur) {
    const ctx = audio.ctx, { buf, rate } = pluckBuffer(freq);
    const src = ctx.createBufferSource(), g = ctx.createGain();
    src.buffer = buf; src.playbackRate.value = rate;
    g.gain.setValueAtTime(1.6, t); g.gain.setValueAtTime(1.6, t + Math.max(dur, 0.25));
    g.gain.linearRampToValueAtTime(0, t + Math.max(dur, 0.25) + 0.25);
    src.connect(g).connect(audio.out); src.start(t); src.stop(t + Math.max(dur, 0.25) + 0.3);
    return src;
  } },
  tone: { label: 'Sustained tone (synthesized)', start(freq, t, dur) {
    const ctx = audio.ctx, o = ctx.createOscillator(), o2 = ctx.createOscillator(), g = ctx.createGain(), g2 = ctx.createGain();
    o.type = 'triangle'; o.frequency.value = freq; o2.type = 'sine'; o2.frequency.value = freq * 2; g2.gain.value = 0.25;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.5, t + 0.03);
    g.gain.setValueAtTime(0.5, t + Math.max(dur - 0.04, 0.04)); g.gain.linearRampToValueAtTime(0, t + dur + 0.03);
    o.connect(g); o2.connect(g2).connect(g); g.connect(audio.out);
    o.start(t); o2.start(t); o.stop(t + dur + 0.05); o2.stop(t + dur + 0.05);
    return o;
  } },
};

const player = { on: false, events: [], next: 0, at: 0, t0: 0, timer: 0, raf: 0, live: [], cursor: null, lit: null };

function playbackEvents(from, bpm) {
  /* One event per symbol, so the cursor visits tied notes too; a tied note
     extends the sound of the note before it instead of starting a new one. */
  const ev = []; let t = 0, lastSound = null;
  const ms = state.piece.measures;
  for (let m = from.m; m < ms.length; m++) {
    for (let i = m === from.m ? from.i : 0; i < ms[m].n.length; i++) {
      const n = ms[m].n[i], dur = seconds(n, bpm);
      const e = { m, i, t, dur, sound: 0 };
      if (!n.rest) {
        if (n.tie && lastSound) lastSound.sound += dur;
        else { e.sound = dur; e.freq = frequency(n); lastSound = e; }
      } else lastSound = null;
      ev.push(e); t += dur;
    }
  }
  return ev;
}

function play() {
  stop();
  const ctx = audioContext();
  if (ctx.state === 'suspended') ctx.resume();
  const bpm = Math.min(300, Math.max(30, +document.getElementById('tempo').value || 100));
  const inst = INSTRUMENTS[document.getElementById('instrument').value];
  player.events = playbackEvents(state.sel || { m: 0, i: 0 }, bpm);
  player.next = 0; player.at = 0; player.t0 = ctx.currentTime + 0.12; player.on = true;
  const schedule = () => {                     // start notes a little ahead of the audio clock
    while (player.next < player.events.length && player.events[player.next].t < ctx.currentTime - player.t0 + 0.2) {
      const e = player.events[player.next++];
      if (e.sound) player.live.push(inst.start(e.freq, player.t0 + e.t, e.sound));
    }
    if (player.live.length > 64) player.live.splice(0, player.live.length - 64);
  };
  schedule(); player.timer = setInterval(schedule, 25);
  if (!player.cursor) { player.cursor = document.createElement('div'); player.cursor.id = 'cursor'; }
  state.host.appendChild(player.cursor);
  document.getElementById('play').textContent = 'Stop';
  const tick = () => {
    if (!player.on) return;
    const now = ctx.currentTime - player.t0, ev = player.events;
    while (player.at + 1 < ev.length && ev[player.at + 1].t <= now) player.at++;
    const last = ev[ev.length - 1];
    if (now > last.t + last.dur) { stop(); return; }
    if (now >= 0) follow(ev[player.at], ev[player.at + 1], now);
    player.raf = requestAnimationFrame(tick);
  };
  player.raf = requestAnimationFrame(tick);
}

function noteElement(e) {
  return state.host.querySelector(`.hit[data-m="${e.m}"][data-i="${e.i}"]`);
}
function follow(e, nextE, now) {
  /* Highlight the sounding note and move the cursor. Only a class and a position
     change; nothing is redrawn unless playback reaches a system that is not in view. */
  const si = systemOf(e.m), sc = state.host.parentElement;
  if (player.lit !== e) {
    if (si * SYS_H < sc.scrollTop || (si + 1) * SYS_H > sc.scrollTop + sc.clientHeight) {
      sc.scrollTop = Math.max(0, si * SYS_H - 20); refresh();
    }
    state.host.querySelectorAll('.playing').forEach(x => x.classList.remove('playing'));
    const el = noteElement(e);
    if (el) el.classList.add('playing');
    const hx = state.host.getBoundingClientRect().x;
    const head = x => { const h = x && (x.querySelector('.vf-notehead') || x); const b = h.getBoundingClientRect(); return b.x + b.width / 2 - hx; };
    e.x = el ? head(el) : 0;
    const nel = nextE && systemOf(nextE.m) === si ? noteElement(nextE) : null;
    e.x2 = nel ? head(nel) : e.x + 24;
    player.lit = e;
    player.cursor.style.top = si * SYS_H + 14 + 'px';
  }
  const f = Math.min(1, Math.max(0, (now - e.t) / e.dur));
  player.cursor.style.transform = `translateX(${e.x + (e.x2 - e.x) * f}px)`;
}

function stop() {
  if (!player.on && !player.cursor) return;
  player.on = false; clearInterval(player.timer); cancelAnimationFrame(player.raf);
  for (const s of player.live) { try { s.stop(); } catch (_) {} }
  player.live = []; player.lit = null;
  if (state.host) state.host.querySelectorAll('.playing').forEach(x => x.classList.remove('playing'));
  if (player.cursor) player.cursor.remove();
  const b = document.getElementById('play'); if (b) b.textContent = 'Play';
}

// ---------------------------------------------------------------- page
const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const frequencyText = n => n.rest ? '' : frequency(n).toFixed(1) + ' Hz';
const fmt = x => x.toFixed(x < 10 ? 1 : 0) + ' ms';

async function start() {
  await document.fonts.ready;
  state.host = document.getElementById('score');
  const sel = document.getElementById('piece'), status = document.getElementById('status');
  for (const p of PIECES) sel.add(new Option(`${p.makam} ${p.form}, ${p.usul} · ${p.notes.toLocaleString()} symbols`, p.name));
  const say = t => { status.textContent = t; };
  const open = async () => {
    stop();
    const p = JSON.parse(JSON.stringify(PIECES.find(q => q.name === sel.value)));
    const r = load(p, document.getElementById('virtual').checked);
    await frame();
    say(`Loaded ${p.notes.toLocaleString()} symbols in ${p.measures.length} measures: widths ${fmt(r.widths_ms)}, ` +
        `line breaks ${fmt(r.breaks_ms)}, drawing ${r.systems_drawn} of ${r.systems} systems ${fmt(r.draw_ms)}.`);
  };
  sel.onchange = open; document.getElementById('virtual').onchange = open;
  document.getElementById('accmode').onchange = e => { state.accMode = e.target.value; open(); };
  state.host.parentElement.addEventListener('scroll', () => refresh(), { passive: true });
  state.host.addEventListener('click', e => {
    const el = e.target.closest('.hit');
    if (!el) return;
    select(+el.dataset.m, +el.dataset.i);
    const n = state.piece.measures[state.sel.m].n[state.sel.i];
    say(`Selected measure ${state.sel.m + 1}, note ${state.sel.i + 1}: ${n.rest ? 'rest' : n.k.replace('/', '')}` +
        (spelled(n) ? ', ' + accName(spelled(n)) : '') + `, ${frequencyText(n)}. Arrows move it; 1–9 flat, Shift+1–9 sharp, 0 natural; I inserts, Delete removes.`);
  });
  document.addEventListener('keydown', e => {
    if (e.target.matches('input, select')) return;
    if (e.key === ' ') { e.preventDefault(); player.on ? stop() : play(); return; }
    if (!state.sel) return;
    const digit = /^Digit[0-9]$/.test(e.code) ? +e.code.slice(5) : -1;
    let r = null;
    if (e.key === 'ArrowUp') r = ops.step(1);
    else if (e.key === 'ArrowDown') r = ops.step(-1);
    else if (digit > 0) r = ops.accidental((e.shiftKey ? '#' : 'b') + digit);
    else if (digit === 0) r = ops.accidental(null);
    else if (e.key === 'i' || e.key === 'I') r = ops.insert();
    else if (e.key === 'Delete' || e.key === 'Backspace') r = ops.remove();
    else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      const d = e.key === 'ArrowRight' ? 1 : -1; let { m, i } = state.sel; i += d;
      if (i < 0 && m > 0) { m--; i = state.piece.measures[m].n.length - 1; }
      else if (i >= state.piece.measures[m].n.length && m + 1 < state.piece.measures.length) { m++; i = 0; }
      if (i >= 0 && i < state.piece.measures[m].n.length) select(m, i);
    } else return;
    e.preventDefault();
    if (r) stop();
    if (r) say(`Edit redrawn in ${fmt(r.ms)} (${r.kind}).`);
  });
  const inst = document.getElementById('instrument');
  for (const [k, v] of Object.entries(INSTRUMENTS)) inst.add(new Option(v.label, k));
  document.getElementById('play').onclick = () => (player.on ? stop() : play());
  drawGlyphTable();
  await open();
  window.ready = true;
}

function drawGlyphTable() {
  /* Both notations on notes and in a key signature: the eight AEU signs, then numbered commas. */
  const div = document.getElementById('glyphs');
  const r = new Renderer(div, Renderer.Backends.SVG); r.resize(1000, 270);
  const ctx = r.getContext(), keep = state.accMode;
  const row = (mode, y, sig, list) => {
    state.accMode = mode;
    const stave = new Stave(10, y, 980).addClef('treble');
    stave.addModifier(new KeySignature('C').setOrdered(sig.map(([a, p]) => ({ type: glyph(a), pitch: p }))));
    stave.addTimeSignature('10/8').setContext(ctx).draw();
    const notes = list.map(([a, k]) => new StaveNote({ keys: [k], duration: 'q', autoStem: true }).addModifier(new Accidental(glyph(a)), 0));
    const v = new Voice({ numBeats: list.length, beatValue: 4 }).addTickables(notes);
    new Formatter().joinVoices([v]).formatToStave([v], stave); v.draw(ctx, stave);
  };
  row('aeu', 10, [['b1', 'b/4'], ['b4', 'e/5'], ['#4', 'f/5']],
      [['b1', 'b/4'], ['b4', 'e/5'], ['b5', 'a/4'], ['b8', 'd/5'], ['#1', 'f/4'], ['#4', 'c/5'], ['#5', 'g/4'], ['#8', 'f/5']]);
  row('numbered', 140, [['b2', 'b/4'], ['#3', 'f/5']],
      [['b1', 'b/4'], ['b2', 'b/4'], ['b3', 'e/5'], ['b4', 'e/5'], ['b5', 'a/4'], ['#1', 'f/5'], ['#2', 'f/5'], ['#3', 'f/5'], ['#4', 'c/5'], ['#5', 'c/5']]);
  state.accMode = keep;
  document.getElementById('glyphnames').textContent =
    'Top: the eight AEU signs (flats of 1, 4, 5, 8 commas, then sharps). Bottom: numbered commas, flats 1 to 5 and sharps 1 to 5. One comma is 22.64 cents.';
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
