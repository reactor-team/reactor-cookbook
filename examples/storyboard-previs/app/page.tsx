'use client';
import { useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowUp, Terminal, Sparkles, Timer, Radio, Clapperboard, Download, Film, ImagePlus, Layers, LoaderCircle, Maximize2, Minimize2, Play, Plus, RotateCcw, Square, Trash2, Volume2, X } from 'lucide-react';
import { loadReferenceKit, kitStory } from '@/lib/reference-kit';
import { shrinkImage } from '@/lib/image';
import { GenerationOverlay } from '@/components/generation-overlay';
import { readRecordings, saveRecording, recordingMime, type Recording } from '@/lib/recordings';
import type { Reactor, FileRef } from '@reactor-team/js-sdk';
import { Board, Shot, Reference, StoryId, desertSignal, storyBoards, stories, durations, durationLabel, moveShot, readReferences, saveReference, deleteReference } from '@/lib/storyboard';
type SavedTake = Recording & { url: string };
type Asset = Reference & { url: string };
type Take = { clipId: string; status: 'queued' | 'ready' | 'playing' | 'played' | 'failed' | 'expired'; prompt: string; buildSeconds?: number };
type Clip = { clip_id: string; metadata: string; prompt: string; build_seconds?: number | null };
const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);
const fmt = (n: number) => n.toFixed(1) + 's';
const now = () => Date.now();
const clock = (): number => performance.now();
export default function Studio() {
  const [board, setBoard] = useState<Board>({ title: 'Desert signal', shots: [] });
  const [loaded, setLoaded] = useState(false);
  const [selected, setSelected] = useState('');
  const [assets, setAssets] = useState<Asset[]>([]);
  const [tab, setTab] = useState<'shots' | 'references'>('shots');
  const [status, setStatus] = useState('disconnected');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [takes, setTakes] = useState<Record<string, Take>>({});
  const [recordings, setRecordings] = useState<Record<string, SavedTake>>({});
  const [replay, setReplay] = useState<{ ids: string[]; position: number } | null>(null);
  const [phase, setPhase] = useState('');
  const [activeShot, setActiveShot] = useState('');
  const [present, setPresent] = useState(false);
  const [live, setLive] = useState(false);
  const [nextPrompt, setNextPrompt] = useState('');
  const [nextMode, setNextMode] = useState<'continue' | 'cut'>('continue');
  const [preloading, setPreloading] = useState(false);
  const [picturesReady, setPicturesReady] = useState(0);
  const nextInput = useRef<HTMLInputElement>(null);
  const [sessionStart, setSessionStart] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  const slotA = useRef<HTMLVideoElement>(null);
  const slotB = useRef<HTMLVideoElement>(null);
  const [log, setLog] = useState<{ t: number; text: string; kind: 'info' | 'error' }[]>([]);
  const [showLog, setShowLog] = useState(false);
  const note = (text: string, kind: 'info' | 'error' = 'info') => { setLog(l => [...l.slice(-299), { t: now(), text, kind }]); if (kind === 'error') console.warn('[previs]', text); else console.info('[previs]', text); };
  const capture = useRef<{ shotId: string; finish: () => void; cancel: (reason: string) => void } | null>(null);
  const lastPlayed = useRef('');
  const sequence = useRef<{ ids: string[]; start: (id: string) => void; end: (id: string) => void; cancel: (reason: string) => void } | null>(null);
  const [playing, setPlaying] = useState('');
  const [sound, setSound] = useState(true);
  const client = useRef<Reactor | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const audio = useRef<HTMLAudioElement>(null);
  const drag = useRef('');
  const assetUrls = useRef<string[]>([]);
  const readyClips = useRef(new Map<string, string>());
  const buildStart = useRef(new Map<string, number>());
  const uploads = useRef(new Map<string, Promise<FileRef>>());
  const assetsRef = useRef(assets); assetsRef.current = assets;
  const pending = useRef(new Map<string, { promise: Promise<string>; resolve: (v: string) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }>());
  const connecting = useRef(false);
  const boardRef = useRef(board); boardRef.current = board;
  const shot = board.shots.find(s => s.id === selected) ?? board.shots[0];
  const index = board.shots.findIndex(s => s.id === shot?.id);
  const take = shot ? takes[shot.id] : undefined;
  const ready = status === 'ready';
  const buildFor = (id: string) => recordings[id]?.buildSeconds ?? takes[id]?.buildSeconds;
  const timelineShots = board.shots.filter(s => !s.added);
  const addedShots = board.shots.filter(s => s.added);
  const filmSeconds = timelineShots.reduce((n, s) => n + s.seconds, 0);
  const builtShots = timelineShots.filter(s => buildFor(s.id) !== undefined);
  const buildTotal = builtShots.reduce((n, s) => n + (buildFor(s.id) ?? 0), 0);
  useEffect(() => {
    const story = (localStorage.getItem('reactor-story') as StoryId | null) ?? 'desert';
    const initial = readSavedBoard(story) ?? storyBoards[story]();
    setBoard(initial); setSelected(initial.shots[0].id); setLoaded(true);
    let cancelled = false;
    readReferences().then(async refs => { if (!localStorage.getItem('reactor-reference-kit-v8')) { for (const stale of ['office-building', 'office-blue', 'office-denim', 'office-navy', 'drop-hq', 'drop-contact', 'drop-station']) { await deleteReference(stale).catch(() => {}); refs = refs.filter(r => r.id !== stale); } const kit = await loadReferenceKit(); for (const ref of kit) await saveReference(ref); localStorage.setItem('reactor-reference-kit-v8', '1'); const ids = new Set(refs.map(r => r.id)); refs = [...refs, ...kit.filter(r => !ids.has(r.id))]; } return refs; }).then(refs => { if (cancelled) return; setAssets(refs.filter(r => r.kind === 'image').map(ref => { const url = URL.createObjectURL(ref.blob); assetUrls.current.push(url); return { ...ref, url }; })); }).catch(() => setError('Reference storage is unavailable in this browser.'));
    readRecordings().then(rows => { if (cancelled) return; setRecordings(Object.fromEntries(rows.map(row => { const url = URL.createObjectURL(row.blob); assetUrls.current.push(url); return [row.id, { ...row, url }]; }))); }).catch(() => setError('Saved takes could not be loaded.'));
    const onFullscreen = () => { if (!document.fullscreenElement) setPresent(false); };
    document.addEventListener('fullscreenchange', onFullscreen);
    return () => { cancelled = true; document.removeEventListener('fullscreenchange', onFullscreen); capture.current?.cancel('Session closed'); void client.current?.disconnect(); assetUrls.current.forEach(URL.revokeObjectURL); pending.current.forEach(p => { clearTimeout(p.timer); p.reject(new Error('Session closed')); }); pending.current.clear(); };
  }, []);
  useEffect(() => { if (!notice) return; const id = setTimeout(() => setNotice(''), 7000); return () => clearTimeout(id); }, [notice]);
  useEffect(() => { if (!sessionStart) return; const id = setInterval(() => setTick(t => t + 1), 1000); return () => clearInterval(id); }, [sessionStart]);
  useEffect(() => { if (loaded) try { const story = board.story ?? 'desert'; localStorage.setItem(`reactor-storyboard-v21:${story}`, JSON.stringify(board)); localStorage.setItem('reactor-story', story); } catch { setError('Browser storage is full.'); } }, [board, loaded]);
  function update(patch: Partial<Shot>) { if (shot) setBoard(b => ({ ...b, shots: b.shots.map(s => s.id === shot.id ? { ...s, ...patch } : s) })); }
  function reorder(source: string, target: string) { if (source === target) return; setBoard(b => ({ ...b, shots: moveShot(b.shots, source, target) })); setNotice('Order updated. Regenerate linked shots to follow the new sequence.'); }
  function addShot() { const s: Shot = { id: crypto.randomUUID(), title: 'New shot', prompt: '', seconds: 8, image: '', refs: shot ? [...shot.refs] : [], connected: false }; setBoard(b => ({ ...b, shots: [...b.shots, s] })); setSelected(s.id); }
  function readSavedBoard(id: StoryId): Board | null { try { const saved = JSON.parse(localStorage.getItem(`reactor-storyboard-v21:${id}`) || 'null'); if (saved && typeof saved.title === 'string' && Array.isArray(saved.shots) && saved.shots.length && saved.shots.every((s: Shot) => typeof s.id === 'string' && typeof s.prompt === 'string' && Array.isArray(s.refs) && durations.includes(s.seconds))) return { ...saved, story: id }; } catch {} return null; }
  // Switching stories keeps each one's edits; Reset script returns the current story to its original script.
  function loadStory(id: StoryId, fresh = false) { const b = (fresh ? null : readSavedBoard(id)) ?? storyBoards[id](); setBoard(b); setSelected(b.shots[0].id); setReplay(null); setLive(false); void restoreKit(); note(fresh ? `Reset script: ${b.title}` : `Story: ${b.title}`); }
  function resetBoard() { loadStory(board.story ?? 'desert', true); }
  function clearAdded() { const keep = board.shots.filter(s => !s.added); if (keep.length === board.shots.length) return; setBoard(b => ({ ...b, shots: b.shots.filter(s => !s.added) })); setSelected(keep[keep.length - 1].id); setReplay(null); setLive(false); setNotice(`Removed ${addedCount} added beat${addedCount === 1 ? '' : 's'}. The original script is back.`); }
  async function upload(files: FileList | null) {
    if (!files) return; setError('');
    for (const file of Array.from(files)) {
      try {
        if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new Error(`${file.name}: use a JPEG, PNG, or WebP image.`);
        if (file.size > 25 * 1024 * 1024) throw new Error(`${file.name}: maximum file size is 25 MB.`);
        const bitmap = await createImageBitmap(file); const ratio = bitmap.width / bitmap.height; const pixels = bitmap.width * bitmap.height; bitmap.close();
        if (pixels > 25000000 || ratio < .25 || ratio > 4) throw new Error(`${file.name}: use at most 25 million pixels and an aspect between 1:4 and 4:1.`);
        const ref: Reference = { id: crypto.randomUUID(), name: file.name, kind: 'image', blob: file };
        await saveReference(ref); const url = URL.createObjectURL(file); assetUrls.current.push(url); setAssets(a => [...a, { ...ref, url }]);
      } catch (e) { setError(errorText(e)); }
    }
    setTab('references');
  }
  async function restoreKit() { try { const kit = await loadReferenceKit(); for (const ref of kit) await saveReference(ref); setAssets(previous => { const ids = new Set(previous.map(a => a.id)); return [...previous, ...kit.filter(r => !ids.has(r.id)).map(r => { const url = URL.createObjectURL(r.blob); assetUrls.current.push(url); return { ...r, url }; })]; }); } catch (e) { setError(errorText(e)); } }
  async function removeAsset(a: Asset) { try { await deleteReference(a.id); uploads.current.delete(a.id); setAssets(v => v.filter(x => x.id !== a.id)); setBoard(b => ({ ...b, shots: b.shots.map(s => ({ ...s, refs: s.refs.filter(id => id !== a.id) })) })); URL.revokeObjectURL(a.url); } catch (e) { setError(errorText(e)); } }
  function handleClip(type: string, clip: Clip, reason?: string) {
    const id = clip.metadata; if (!id) return;
    const state: Take['status'] = type === 'clip_generated' ? 'ready' : type === 'clip_started' ? 'playing' : type === 'clip_failed' ? 'failed' : type === 'clip_finished' || type === 'clip_stopped' ? 'played' : 'queued';
    // Prefer the model's own build time; fall back to the enqueue-to-ready interval measured here.
    const started = buildStart.current.get(id);
    const buildSeconds = type === 'clip_generated' ? (typeof clip.build_seconds === 'number' ? clip.build_seconds : started ? (now() - started) / 1000 : undefined) : undefined;
    setTakes(t => ({ ...t, [id]: { clipId: clip.clip_id, status: state, prompt: clip.prompt, buildSeconds: buildSeconds ?? t[id]?.buildSeconds } }));
    const title = boardRef.current.shots.find(x => x.id === id)?.title ?? id;
    if (type === 'clip_generated') note(`Generated “${title}”${buildSeconds !== undefined ? ` in ${buildSeconds.toFixed(1)}s` : ''}`); else if (type === 'clip_failed') note(`Failed “${title}”: ${reason || 'no reason given'}`, 'error'); else if (type !== 'clip_queued') note(`${type.replace('clip_', 'Clip ')}: “${title}”`);
    if (type === 'clip_generated') { readyClips.current.set(id, clip.clip_id); const p = pending.current.get(id); if (p) { clearTimeout(p.timer); p.resolve(clip.clip_id); pending.current.delete(id); } }
    const seq = sequence.current;
    if (seq && seq.ids.includes(id)) {
      if (type === 'clip_started') { lastPlayed.current = id; setPlaying(id); setSelected(id); setActiveShot(id); setPhase(''); setNextPrompt(''); seq.start(id); return; }
      if (type === 'clip_finished') { seq.end(id); return; }
      if (type === 'clip_stopped') { seq.cancel('Playback stopped before the sequence finished.'); setPlaying(''); return; }
      if (type === 'clip_failed') seq.cancel(reason || 'Generation failed.');
    }
    if (type === 'clip_started') { lastPlayed.current = id; setPlaying(id); }
    if (type === 'clip_finished') { if (capture.current?.shotId === id) capture.current.finish(); else setPlaying(''); }
    if (type === 'clip_stopped') { capture.current?.cancel('Playback stopped before this take was saved.'); setPlaying(''); }
    if (type === 'clip_failed') { const p = pending.current.get(id); if (p) { clearTimeout(p.timer); p.reject(new Error(reason || 'Generation failed')); pending.current.delete(id); } setError(reason || 'Generation failed.'); }
  }
  function ensureAudio() { const a = audio.current; if (!a) return; a.muted = !sound; void a.play().catch(() => { const retry = () => { a.muted = !sound; void a.play().catch(() => {}); }; document.addEventListener('pointerdown', retry, { once: true }); }); }
  async function connect() {
    if (connecting.current) return; connecting.current = true; setError(''); setStatus('connecting'); note('Requesting session token');
    try {
      await client.current?.disconnect(); readyClips.current.clear(); uploads.current.clear(); setTakes({});
      const response = await fetch('/api/token', { method: 'POST' }); const data = await response.json() as { jwt: string; error?: string }; if (!response.ok) throw new Error(data.error);
      const { Reactor } = await import('@reactor-team/js-sdk');
      const r = new Reactor({ modelName: 'reactor/fast-h3-omni' }); client.current = r;
      r.on('statusChanged', s => { setStatus(s); note(`Connection: ${s}`); if (s === 'ready') { setSessionStart(prev => prev ?? now()); void r.sendCommand('set_flush_on_clip_end', { enabled: false }).catch(() => {}); void preloadUploads(); } if (s === 'disconnected') { setSessionStart(null); setPicturesReady(0); setPreloading(false); } if (s === 'disconnected') { setLive(false); readyClips.current.clear(); uploads.current.clear(); capture.current?.cancel('Session disconnected before capture finished.'); setPlaying(''); setTakes(t => Object.fromEntries(Object.entries(t).map(([id, take]) => [id, { ...take, status: 'expired' }]))); pending.current.forEach(p => { clearTimeout(p.timer); p.reject(new Error('Session disconnected')); }); pending.current.clear(); } });
      r.on('trackReceived', (name, _track, stream) => { if (name === 'main_video' && video.current) { video.current.srcObject = stream; void video.current.play().catch(() => setNotice('Press Play in the video controls to enable playback.')); } if (name === 'main_audio' && audio.current) { audio.current.srcObject = stream; audio.current.muted = !sound; void audio.current.play().catch(() => { note('Browser blocked audio until the next click; it will start on your next action.'); const retry = () => { if (audio.current) { audio.current.muted = !sound; void audio.current.play().catch(() => {}); } }; document.addEventListener('pointerdown', retry, { once: true }); }); } });
      r.on('error', e => { setError(e.message); note(`SDK error: ${e.message}`, 'error'); });
      r.on('message', message => { const d = message.data as { clip?: Clip; reason?: string; generation?: Clip[]; playout?: Clip[] }; if (d?.clip) handleClip(message.type, d.clip, d.reason); if (message.type === 'queue_update') for (const c of [...(d.generation ?? []), ...(d.playout ?? [])]) if (typeof c.build_seconds === 'number' && c.metadata) setTakes(t => t[c.metadata] ? { ...t, [c.metadata]: { ...t[c.metadata], buildSeconds: c.build_seconds as number } } : t); if (message.type === 'command_error') { setError(d.reason || 'Command refused.'); note(`Command refused: ${d.reason || 'no reason'}`, 'error'); } });
      await r.connect(data.jwt);
    } catch (e) { setError(errorText(e)); note(`Connect failed: ${errorText(e)}`, 'error'); setStatus('disconnected'); await client.current?.disconnect(); client.current = null; } finally { connecting.current = false; }
  }
  async function command(name: string, data: Record<string, unknown> = {}) { if (!client.current) throw new Error('Connect to H3 Omni first.'); const reply = await client.current.sendCommand(name, data); if (reply?.type === 'command_error') throw new Error((reply.data as { reason: string }).reason); return reply; }
  function waitForClip(id: string): Promise<string> {
    const known = readyClips.current.get(id); if (known) return Promise.resolve(known);
    const existing = pending.current.get(id); if (existing) return existing.promise;
    let resolve!: (v: string) => void, reject!: (e: Error) => void;
    const promise = new Promise<string>((res, rej) => { resolve = res; reject = rej; });
    const timer = setTimeout(() => { pending.current.delete(id); reject(new Error('Generation is taking longer than expected. Check the shot status before retrying.')); }, 300000);
    pending.current.set(id, { promise, resolve, reject, timer });
    return promise;
  }
  function uploadAsset(a: Asset): Promise<FileRef> {
    const cached = uploads.current.get(a.id); if (cached) return cached;
    const job = (async () => {
      const small = await shrinkImage(a.blob);
      const safeName = (a.name.normalize('NFKD').replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'picture') + '.jpg';
      note(`Uploading ${safeName}: ${(a.blob.size / 1024).toFixed(0)} KB → ${(small.size / 1024).toFixed(0)} KB`);
      const t0 = clock();
      const ref = await client.current!.uploadFile(new File([small], safeName, { type: small.type || 'image/jpeg' }));
      const secs = (clock() - t0) / 1000;
      note(`Uploaded ${safeName} in ${secs.toFixed(1)}s (${(small.size / 1024 / secs).toFixed(0)} KB/s)`);
      return ref;
    })();
    uploads.current.set(a.id, job);
    job.catch(() => uploads.current.delete(a.id));
    return job;
  }
  // Upload every picture the storyboard uses as soon as the session is ready, so the first enqueue has nothing to wait for.
  async function preloadUploads() {
    const ids = new Set(boardRef.current.shots.flatMap(s => s.refs));
    const refs = assetsRef.current.filter(a => ids.has(a.id));
    if (!refs.length) return;
    note(`Preloading ${refs.length} pictures`); setPreloading(true); setPicturesReady(0);
    try { for (const a of refs) await uploadAsset(a); setPicturesReady(refs.length); note('Pictures ready'); } catch (e) { note(`Preload failed: ${errorText(e)}`, 'error'); } finally { setPreloading(false); }
  }
  async function enqueue(s: Shot, previous?: Shot) {
    if (!s.prompt.trim()) throw new Error(`Add direction for “${s.title}”.`);
    const refs = s.refs.map(id => assets.find(a => a.id === id)).filter((a): a is Asset => !!a);
    if (refs.length > 9) throw new Error('Use up to 9 reference pictures per shot.');
    let continuation: string | undefined;
    if (s.connected) { if (!previous) throw new Error('The first shot cannot continue a previous shot.'); continuation = readyClips.current.get(previous.id); if (!continuation) throw new Error('Generate the previous shot first, or generate the whole storyboard.'); }
    setPhase('Uploading references');
    const uploaded: FileRef[] = [];
    for (const [i, a] of refs.entries()) {
      if (!uploads.current.has(a.id)) setPhase(`Uploading picture ${i + 1} of ${refs.length}`);
      uploaded.push(await uploadAsset(a));
    }
    const payload: Record<string, unknown> = { prompt: s.prompt, seconds: s.seconds, metadata: s.id };
    if (uploaded.length) payload.reference_images = uploaded;
    if (continuation) { payload.continue_from_clip_id = continuation; payload.continue_mode = s.continueMode ?? 'latent'; }
    readyClips.current.delete(s.id);
    buildStart.current.set(s.id, now());
    const t1 = clock(); note(`Enqueue “${s.title}” (${durationLabel(s.seconds)}, ${uploaded.length} pictures${continuation ? `, continues clip ${continuation.slice(0, 8)} in ${s.continueMode ?? 'latent'} mode` : ''})`); let reply; try { reply = await command('enqueue', payload); } catch (e) { if (continuation && payload.continue_mode === 'frame') { note(`Frame continuation refused (${errorText(e)}); retrying in latent mode`, 'error'); payload.continue_mode = 'latent'; reply = await command('enqueue', payload); } else throw e; } note(`Enqueue acknowledged in ${((clock() - t1) / 1000).toFixed(2)}s`); if (reply?.type !== 'clip_queued') throw new Error('Reactor did not confirm this shot.');
    handleClip('clip_queued', (reply.data as { clip: Clip }).clip);
  }
  async function captureShot(s: Shot, clipId: string) {
    const videoStream = video.current?.srcObject as MediaStream | null;
    const audioStream = audio.current?.srcObject as MediaStream | null;
    if (!videoStream?.getVideoTracks().length || !audioStream?.getAudioTracks().length) throw new Error('Video or audio track is not available yet. Wait a moment, then play this ready shot.');
    const stream = new MediaStream([...videoStream.getVideoTracks(), ...audioStream.getAudioTracks()]);
    const recorder = new MediaRecorder(stream, { mimeType: recordingMime(), videoBitsPerSecond: 8000000 });
    setPhase('');
    await new Promise<void>((resolve, reject) => {
      let settled = false, failed = false; let finishTimer: ReturnType<typeof setTimeout> | undefined; const chunks: Blob[] = [];
      const timeout = setTimeout(() => cancel('Playback timed out. Reconnect and generate this shot again.'), (s.seconds + 45) * 1000);
      const cleanup = () => { clearTimeout(timeout); clearTimeout(finishTimer); if (capture.current?.shotId === s.id) capture.current = null; };
      const cancel = (reason: string) => { if (settled) return; settled = true; failed = true; cleanup(); if (recorder.state !== 'inactive') recorder.stop(); reject(new Error(reason)); };
      recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      recorder.onerror = () => cancel('Recording failed. Try generating this shot again.');
      recorder.onstop = async () => { if (failed) return; cleanup(); setPlaying(''); const blob = new Blob(chunks, { type: recorder.mimeType }); if (!blob.size) { settled = true; reject(new Error('No video was captured.')); return; } const recording: Recording = { id: s.id, blob, prompt: s.prompt, seconds: s.seconds, createdAt: now(), buildSeconds: buildFor(s.id) }; const url = URL.createObjectURL(blob); assetUrls.current.push(url); setRecordings(r => ({ ...r, [s.id]: { ...recording, url } })); note(`Saved take “${s.title}” (${(blob.size / 1024 / 1024).toFixed(1)} MB)`); try { await saveRecording(recording); settled = true; resolve(); } catch { settled = true; reject(new Error('Take captured but storage is full. Download it before leaving this page.')); } };
      capture.current = { shotId: s.id, finish: () => { if (finishTimer) return; finishTimer = setTimeout(() => { if (recorder.state !== 'inactive') recorder.stop(); }, 600); }, cancel };
      // Capture both negotiated tracks before consuming the one-time model playback.
      recorder.start(250);
      void command('play', { clip_id: clipId }).then(reply => { if (reply?.type === 'clip_started') handleClip('clip_started', (reply.data as { clip: Clip }).clip); }).catch(e => cancel(errorText(e)));
    });
  }
  async function runSequence(plan: Shot[] = timelineShots, order: Shot[] = board.shots, saveSequence = true): Promise<boolean> {
    setBusy(true); setError(''); setNotice(''); setReplay(null); ensureAudio();
    const ids = plan.map(s => s.id);
    const previousOf = (s: Shot) => order[order.findIndex(x => x.id === s.id) - 1];
    try {
      const mime = recordingMime();
      const videoStream = video.current?.srcObject as MediaStream | null;
      const audioStream = audio.current?.srcObject as MediaStream | null;
      if (!videoStream?.getVideoTracks().length || !audioStream?.getAudioTracks().length) throw new Error('Video or audio track is not available yet. Wait a moment and try again.');
      const stream = new MediaStream([...videoStream.getVideoTracks(), ...audioStream.getAudioTracks()]);
      const main = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8000000 });
      const mainChunks: Blob[] = []; main.ondataavailable = e => { if (e.data.size) mainChunks.push(e.data); };
      const shotRecorders = new Map<string, MediaRecorder>();
      const timers = new Set<ReturnType<typeof setTimeout>>();
      const later = (fn: () => void, ms: number) => { const t = setTimeout(() => { timers.delete(t); fn(); }, ms); timers.add(t); };
      const saveTake = async (recording: Recording) => { const url = URL.createObjectURL(recording.blob); assetUrls.current.push(url); setRecordings(r => ({ ...r, [recording.id]: { ...recording, url } })); try { await saveRecording(recording); } catch { setError('Take captured but storage is full. Download it before leaving this page.'); } };
      const done = new Promise<void>((resolve, reject) => {
        let settled = false;
        const stopAll = () => { timers.forEach(clearTimeout); timers.clear(); shotRecorders.forEach(r => { if (r.state !== 'inactive') r.stop(); }); };
        const cancel = (reason: string) => { if (settled) return; settled = true; stopAll(); if (main.state !== 'inactive') main.stop(); sequence.current = null; reject(new Error(reason)); };
        main.onstop = () => { if (settled) return; settled = true; sequence.current = null; const blob = new Blob(mainChunks, { type: main.mimeType }); if (saveSequence && blob.size) { void saveTake({ id: 'sequence', blob, prompt: boardRef.current.title, seconds: plan.reduce((n, x) => n + x.seconds, 0), createdAt: now(), shots: ids }); note(`Saved the full sequence (${(blob.size / 1024 / 1024).toFixed(1)} MB)`); } resolve(); };
        main.onerror = () => cancel('Recording failed. Try generating the sequence again.');
        sequence.current = {
          ids,
          start: id => {
            if (shotRecorders.has(id)) return;
            const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8000000 }); const chunks: Blob[] = [];
            rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
            rec.onstop = () => { const blob = new Blob(chunks, { type: rec.mimeType }); const x = plan.find(y => y.id === id); if (blob.size && x) { void saveTake({ id, blob, prompt: x.prompt, seconds: x.seconds, createdAt: now(), buildSeconds: buildFor(id) }); note(`Saved take “${x.title}”`); } };
            shotRecorders.set(id, rec); rec.start(250);
          },
          end: id => {
            // Let the encoder flush the last frames before closing this shot's recorder.
            later(() => { const r = shotRecorders.get(id); if (r && r.state !== 'inactive') r.stop(); }, 250);
            const position = ids.indexOf(id);
            if (position === ids.length - 1) later(() => { stopAll(); if (main.state !== 'inactive') main.stop(); }, 350);
            else { const next = plan[position + 1]; if (next && !readyClips.current.has(next.id)) setPhase(`Generating “${next.title}”`); }
          },
          cancel,
        };
        later(() => cancel('The sequence timed out. Reconnect and try again.'), (plan.reduce((n, x) => n + x.seconds, 0) + 90 * plan.length) * 1000);
      });
      main.start(250);
      await command('set_autoplay', { enabled: true });
      setActiveShot(plan[0].id); setSelected(plan[0].id); setPhase(saveSequence ? `Generating “${plan[0].title}”` : plan[0].connected ? `Generating what happens next: “${plan[0].title}”` : `Generating the next shot: “${plan[0].title}”`);
      note(saveSequence ? 'Sequence started: the model plays each shot as soon as it is built' : `Continuing the story: “${plan[0].prompt.slice(0, 60)}”`);
      // Producer: queue every shot; a linked shot waits only for its predecessor's clip id.
      const producer = (async () => { for (const x of plan) { const prev = previousOf(x); if (x.connected && prev) await waitForClip(prev.id); await enqueue(x, prev); } })().catch(e => sequence.current?.cancel(errorText(e)));
      await done; await producer;
      setLive(true); setTimeout(() => nextInput.current?.focus(), 50);
      setNotice(saveSequence ? 'Sequence saved. The picture is holding on the last frame: tell it what happens next.' : 'Added to the storyboard. What happens next?');
      return true;
    } catch (e) { setError(errorText(e)); note(`Sequence stopped: ${errorText(e)}`, 'error'); setTakes(t => Object.fromEntries(Object.entries(t).map(([id, take]) => [id, take.status === 'queued' ? { ...take, status: 'failed' } : take]))); return false; }
    finally { sequence.current = null; void command('set_autoplay', { enabled: false }).catch(() => {}); setBusy(false); setPhase(''); setActiveShot(''); setPlaying(''); }
  }
  // "What happens next?": append a linked shot and play it from the held frame.
  async function continueStory() {
    const text = nextPrompt.trim(); if (!text) return;
    const candidates = [lastPlayed.current, ...board.shots.map(x => x.id).reverse()];
    const sourceId = candidates.find(id => id && readyClips.current.has(id));
    const last = board.shots.find(x => x.id === sourceId);
    if (!last) { setError('Nothing to continue from yet. Generate a shot first.'); note('Continue refused: no generated clip in this session', 'error'); return; }
    ensureAudio();
    // Expand the beat into full direction with the director model; fall back to the raw beat if it is unavailable.
    let direction = text;
    setBusy(true); setPhase('Writing the direction…');
    try {
      const pictureNames = last.refs.map(id => assets.find(a => a.id === id)?.name.replace(/\.[^.]+$/, '') ?? 'picture');
      const res = await fetch('/api/direct', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ beat: text, mode: nextMode, story: { title: board.title, shots: timelineShots.map(x => ({ title: x.title, prompt: x.prompt })) }, source: { title: last.title, prompt: last.prompt }, pictures: pictureNames }) });
      const data = await res.json() as { prompt?: string; error?: string };
      if (res.ok && data.prompt) { direction = data.prompt; } else note(`Director unavailable, using the beat as written: ${data.error || res.status}`, 'error');
    } catch (e) { note(`Director unavailable, using the beat as written: ${errorText(e)}`, 'error'); }
    setPhase(`Generating what happens next: “${text}”`);
    // The model's recipe for a continuation: same pictures in the same order, the same bindings, then the new action and what carries over.
    const bindings = last.prompt.split('\n').filter(l => /^<(Subject|Picture) \d+> is /.test(l));
    const newLines = direction.split('\n').map(l => l.trim()).filter(l => l && !/^<(Subject|Picture) \d+> is /.test(l));
    const action = newLines.length ? newLines : [`In one shot, ${text}`];
    const carry = nextMode === 'continue'
      ? 'This clip continues directly from the final frame of the previous clip. Keep the same people, their appearance, the setting, the light, and the sound from the previous clip, and change only what happens next.'
      : 'This is the next shot in the same scene: keep the same people, their appearance, the setting, and the light from the previous shot.';
    direction = [...bindings, carry, ...action].join('\n');
    note(`Direction: ${direction.replace(/\n/g, ' / ')}`);
    const s: Shot = { id: crypto.randomUUID(), title: text.length > 34 ? text.slice(0, 32).trimEnd() + '…' : text, prompt: direction, seconds: 8, image: last.image, refs: [...last.refs], connected: nextMode === 'continue', continueMode: 'frame', added: true };
    note(`Continue from “${last.title}”`);
    const order = [last, s];
    setBoard(b => ({ ...b, shots: [...b.shots, s] }));
    const ok = await runSequence([s], order, false);
    if (ok) { setNextPrompt(''); setTimeout(() => nextInput.current?.focus(), 50); }
  }
  async function generate(all = false) {
    if (all) return runSequence();
    setBusy(true); setError(''); setNotice(''); setReplay(null);
    try {
      recordingMime(); const plan = all ? board.shots : [shot];
      const previousOf = (s: Shot) => board.shots[board.shots.findIndex(x => x.id === s.id) - 1];
      // Producer: queue every shot up front. A linked shot waits only for its predecessor's clip id, not for playback.
      let producerError: Error | null = null;
      const producer = (async () => { for (const s of plan) { const prev = previousOf(s); if (s.connected && prev) await waitForClip(prev.id); await enqueue(s, prev); } })().catch(e => { producerError = e instanceof Error ? e : new Error(String(e)); });
      // Consumer: play each shot the moment it is built, while later shots keep generating.
      for (const s of plan) {
        if (producerError) throw producerError;
        setActiveShot(s.id); setSelected(s.id);
        if (!readyClips.current.has(s.id)) setPhase(`Generating “${s.title}”`);
        const clipId = await waitForClip(s.id);
        setPhase('');
        await captureShot(s, clipId);
      }
      await producer; if (producerError) throw producerError;
      setNotice(all ? 'Sequence saved. Play it back in order, with or without the model connected.' : 'Take saved. Replay it anytime, even after disconnecting.');
    }
    catch (e) { setError(errorText(e)); note(`Generation stopped: ${errorText(e)}`, 'error'); setTakes(t => Object.fromEntries(Object.entries(t).map(([id, take]) => [id, take.status === 'queued' ? { ...take, status: 'failed' } : take]))); } finally { setBusy(false); setPhase(''); setActiveShot(''); }
  }
  async function playShot() { if (recordings[shot.id]) { setLive(false); setReplay({ ids: [shot.id], position: 0 }); return; } if (!take) return; setBusy(true); setError(''); try { await captureShot(shot, take.clipId); } catch (e) { setError(errorText(e)); } finally { setBusy(false); } }
  const sequenceTake = recordings['sequence'] && recordings['sequence'].shots && timelineShots.slice(0, recordings['sequence'].shots.length).map(s => s.id).join() === recordings['sequence'].shots.join() ? recordings['sequence'] : undefined;
  const replayPlan = (): string[] | string => { const covered = sequenceTake?.shots?.length ?? 0; const rest = timelineShots.slice(covered); const missing = rest.filter(s => !recordings[s.id]); if (missing.length) return `Generate ${missing.length} remaining shot${missing.length === 1 ? '' : 's'} before playing the whole sequence.`; return [...(sequenceTake ? ['sequence'] : []), ...rest.map(s => s.id)]; };
  function playSequence() { setLive(false); const plan = replayPlan(); if (typeof plan === 'string') { setError(plan); return; } setError(''); setSelected(board.shots[0].id); setReplay({ ids: plan, position: 0 }); }
  function nextReplay() { if (!replay) return; const position = replay.position + 1; if (position >= replay.ids.length) { setReplay(null); return; } if (replay.ids[position] !== 'sequence') setSelected(replay.ids[position]); setReplay({ ...replay, position }); }
  function stopPlayback() { if (replay) { setReplay(null); return; } setLive(false); void command('stop').catch(e => setError(errorText(e))); }
  function downloadTake() { const saved = recordings[shot.id]; if (!saved) return; const a = document.createElement('a'); a.href = saved.url; a.download = `${shot.title.replace(/[^a-z0-9]/gi, '-')}.${saved.blob.type.includes('mp4') ? 'mp4' : 'webm'}`; a.click(); }
  function togglePresent() { if (present) { setPresent(false); if (document.fullscreenElement) void document.exitFullscreen().catch(() => {}); return; } setPresent(true); void document.documentElement.requestFullscreen?.().catch(() => {}); }
  const preferredReference = shot?.image?.startsWith('/references/') ? `desert-${shot.image.split('/').pop()?.split('.')[0]}` : undefined;
  const referenceCover = shot?.refs.includes(preferredReference || '') ? assets.find(a => a.id === preferredReference) : shot?.refs.map(id => assets.find(a => a.id === id)).find(a => a?.kind === 'image');
  const replayTake = replay ? recordings[replay.ids[replay.position]] : null;
  const replayNext = replay && replay.position + 1 < replay.ids.length ? recordings[replay.ids[replay.position + 1]] : null;
  const currentSlot = replay ? replay.position % 2 : 0;
  // Start the slot that already holds the current file; the other slot preloads the next one.
  useEffect(() => { if (!replay) return; const v = (currentSlot === 0 ? slotA : slotB).current; if (!v) return; v.currentTime = 0; void v.play().catch(() => setNotice('Press play on the video to start playback.')); }, [replay, currentSlot]);
  const generationShot = board.shots.find(s => s.id === activeShot);
  const generationImage = generationShot?.refs.map(id => assets.find(a => a.id === id)).find(a => a?.kind === 'image')?.url || generationShot?.image || referenceCover?.url || shot?.image;
  const overlayPhase = phase || (status === 'connecting' ? 'Connecting' : status === 'waiting' ? 'Assigning GPU' : '');
  const coverFor = (s: Shot) => { const preferred = s.image?.startsWith('/references/') ? `desert-${s.image.split('/').pop()?.split('.')[0]}` : undefined; return (preferred && s.refs.includes(preferred) ? assets.find(a => a.id === preferred)?.url : undefined) || s.refs.map(id => assets.find(a => a.id === id)).find(a => a?.kind === 'image')?.url || s.image; };
  const shotBuild = shot ? buildFor(shot.id) : undefined;
  const generatingNow = ready ? Object.values(takes).filter(t => t.status === 'queued').length : 0;
  const addedCount = board.shots.filter(s => s.added).length;
  // Only this story's kit plus the user's own uploads; the other story's cast stays out of the way.
  const storyAssets = assets.filter(a => { const st = kitStory(a.id); return !st || st === (board.story ?? 'desert'); });
  return <div className={present ? 'studio presenting' : 'studio'}>
    <header className="topbar"><span className="brand"><img src="/brand/symbol.svg" alt=""/><img src="/brand/wordmark.svg" alt="Reactor"/></span><span className="brand-divider"/><span className="studio-label"><strong>Previs</strong></span><div className="top-actions"><span className="model-badge"><span/> H3 Omni</span><span className={sessionStart ? 'session-clock' : 'session-clock idle'} title="Time since the model session became ready"><Timer size={12}/> {(() => { const s = sessionStart ? Math.max(0, Math.floor((now() - sessionStart) / 1000)) + 0 * tick : 0; return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; })()}</span><button className={ready ? 'connected connect-button' : 'primary connect-button'} disabled={status !== 'disconnected' && !ready && status !== 'error'} onClick={() => ready ? void client.current?.disconnect() : void connect()}>{ready ? 'Disconnect' : status === 'disconnected' || status === 'error' ? <><Radio size={14}/> Connect model</> : <><LoaderCircle className="spin" size={14}/> Connecting…</>}</button></div></header>
    <div className="projectbar"><div className="project-head"><input className="project-title" aria-label="Storyboard title" value={board.title} onChange={e => setBoard(b => ({ ...b, title: e.target.value }))}/><div className="story-switch"><span className="story-switch-label">Story</span><div className="story-picker" role="tablist" aria-label="Story">{stories.map(st => <button key={st.id} role="tab" aria-selected={(board.story ?? 'desert') === st.id} className={(board.story ?? 'desert') === st.id ? 'active' : ''} disabled={busy} onClick={() => { if ((board.story ?? 'desert') !== st.id) loadStory(st.id); }}>{st.name}</button>)}</div></div></div><div className="project-actions"><button disabled={busy || !!playing || typeof replayPlan() === 'string'} onClick={playSequence}><Play size={15}/> Play sequence</button><button className="primary" disabled={!ready || busy || preloading || !timelineShots.length} onClick={() => void generate(true)}>{busy || preloading ? <LoaderCircle className="spin" size={15}/> : <Clapperboard size={15}/>} {preloading ? 'Preparing pictures…' : 'Generate storyboard'}</button></div></div>
    {(error || notice) && <div className={error ? 'message error' : 'message'} role={error ? 'alert' : 'status'}><span>{error || notice}</span><button aria-label="Dismiss message" onClick={() => { setError(''); setNotice(''); }}><X size={15}/></button></div>}
    <main className="workspace"><aside className="sidebar"><div className="side-tabs"><button className={tab === 'shots' ? 'active' : ''} onClick={() => setTab('shots')}><Clapperboard size={15}/> Shots <span>{board.shots.length}</span></button><button className={tab === 'references' ? 'active' : ''} onClick={() => setTab('references')}><Layers size={15}/> References</button></div>
      {tab === 'shots' ? <><div className="side-heading"><span>Shot list</span><span className="side-actions">{addedCount > 0 && <button onClick={clearAdded} disabled={busy}><X size={11}/> Clear {addedCount} added</button>}<button onClick={resetBoard} disabled={busy}><RotateCcw size={11}/> Reset script</button></span></div><div className="shot-list">{timelineShots.map((s, i) => <div key={s.id} className={`shot-card ${s.id === shot?.id ? 'selected' : ''}`} draggable={!busy} onDragStart={() => { drag.current = s.id; }} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); reorder(drag.current, s.id); }}><button className="shot-select" onClick={() => setSelected(s.id)}><div className="shot-thumb">{coverFor(s) ? <img src={coverFor(s)} alt=""/> : <Film size={22}/>}<span>{String(i + 1).padStart(2, '0')}</span></div><div className="shot-copy"><strong>{s.title}</strong><small>{durationLabel(s.seconds)} <span>•</span> {buildFor(s.id) !== undefined ? `built in ${fmt(buildFor(s.id)!)}` : recordings[s.id] ? 'Saved take' : takes[s.id]?.status || 'Draft'}{s.added ? ' • Added live' : s.connected ? ' • Linked' : ''}</small></div></button><div className="reorder"><button aria-label={`Move ${s.title} up`} disabled={i === 0 || busy} onClick={() => reorder(s.id, board.shots[i - 1].id)}><ArrowUp size={12}/></button><button aria-label={`Move ${s.title} down`} disabled={i === board.shots.length - 1 || busy} onClick={() => reorder(s.id, board.shots[i + 1].id)}><ArrowDown size={12}/></button></div></div>)}</div><button className="add-shot" onClick={addShot}><Plus size={16}/> Add shot</button>{addedShots.length > 0 && <><div className="side-heading added-heading"><span>Added live</span><span>{addedShots.length}</span></div><div className="shot-list">{addedShots.map((s, i) => <div key={s.id} className={`shot-card added ${s.id === shot?.id ? 'selected' : ''}`}><button className="shot-select" onClick={() => setSelected(s.id)}><div className="shot-thumb">{coverFor(s) ? <img src={coverFor(s)} alt=""/> : <Film size={22}/>}<span>+{i + 1}</span></div><div className="shot-copy"><strong>{s.title}</strong><small>{durationLabel(s.seconds)} <span>•</span> {recordings[s.id] ? 'Saved take' : takes[s.id]?.status || 'Draft'}{s.connected ? ` • Continues (${s.continueMode ?? 'latent'})` : ' • Cut'}</small></div></button></div>)}</div></>}<div className="sidebar-foot"><span>{timelineShots.length} shots</span><span>{fmt(filmSeconds)} of film</span></div></> : <><div className="side-heading"><span>Shared by every shot</span></div><label className="upload-area"><ImagePlus size={24}/><strong>Add a picture</strong><small>Actor, location, or prop · JPEG, PNG, WebP</small><input type="file" multiple accept="image/jpeg,image/png,image/webp" onChange={e => { void upload(e.target.files); e.target.value = ''; }}/></label><p className="reference-help">Every shot names its pictures as &lt;Picture 1&gt;, &lt;Picture 2&gt;… in the order they are selected. The same face, place, and prop stay consistent across the whole sequence.</p><div className="asset-list">{storyAssets.map(a => <div className="asset" key={a.id}><img src={a.url} alt={a.name}/><span>{a.name}<small>{a.name.toLowerCase().startsWith('character') ? 'cast' : a.name.toLowerCase().startsWith('location') ? 'location' : a.name.toLowerCase().startsWith('prop') ? 'prop' : 'picture'}</small></span><button aria-label={`Remove ${a.name}`} onClick={() => void removeAsset(a)}><X size={14}/></button></div>)}</div></>}
    </aside>
    <section className="canvas-column"><div className="canvas-heading"><span><Film size={16}/> Shot {String(index + 1).padStart(2, '0')} <span className="muted">/ {String(board.shots.length).padStart(2, '0')}</span></span><span className="draft-label">{playing || replay ? 'PROGRAM' : 'SOURCE'} <span className="monitor-mode">{replay ? 'Sequence playback' : playing ? (sequence.current ? 'Live sequence from H3 Omni' : 'Live from H3 Omni') : live ? 'Holding on the last frame' : 'Storyboard panel'}</span></span></div><div className="viewer" aria-busy={!!overlayPhase}><video ref={video} autoPlay playsInline muted controls={!!playing} className={playing || live ? 'live-video visible' : 'live-video'}/><audio ref={audio} autoPlay muted={!sound || !!replay}/>{replay && [slotA, slotB].map((ref, i) => { const isCurrent = i === currentSlot; const take = isCurrent ? replayTake : replayNext; return <video key={i} ref={ref} className={isCurrent ? 'replay-video' : 'replay-video standby'} src={take?.url} preload="auto" playsInline controls={isCurrent} muted={!isCurrent || !sound} onEnded={isCurrent ? nextReplay : undefined} onError={isCurrent ? () => setError('This saved take could not be played. Download it or generate another take.') : undefined}/>; })} {overlayPhase && !playing && !live && !replay && <GenerationOverlay phase={overlayPhase} image={generationImage} title={generationShot?.title}/>} {!playing && !live && !replay && <div className="concept">{referenceCover || shot?.image ? <img className="concept-image" src={referenceCover?.url || shot?.image} alt={referenceCover ? 'Selected reference' : 'Storyboard reference'}/> : <div className="blank-frame"><Clapperboard size={46}/></div>}<div className="frame-shade"/><span className="frame-tag">Storyboard panel</span><div className="frame-title"><span>SC {String(index + 1).padStart(3, '0')} <span className="monitor-separator">/</span> {board.title}</span><h1>{shot?.title}</h1><p>{shotBuild !== undefined ? `${durationLabel(shot.seconds)} of film, generated in ${fmt(shotBuild)}` : recordings[shot?.id] ? 'Saved take available' : take?.status === 'ready' ? 'Ready to play' : 'Not generated yet'}</p></div><div className="frame-corners"/></div>}{replay && <div className="replay-badge">{replay.ids[0] === 'sequence' ? `Full sequence · ${board.shots.length} shots` : `Shot ${String(replay.position + 1).padStart(2, '0')} / ${String(replay.ids.length).padStart(2, '0')} · ${board.shots.find(s => s.id === replay.ids[replay.position])?.title ?? ''}`}</div>}{!replay && (!!playing || generatingNow > 0) && <div className="player-pills">{!!playing && <span className="pill live"><i/>Live</span>}{generatingNow > 0 && <span className="pill generating"><i/>Generating in real time</span>}</div>}{(!!playing || live) && phase && (phase.startsWith('Generating what') || phase.startsWith('Generating the next') || phase.startsWith('Writing') ? <div className="next-overlay"><GenerationOverlay phase={phase.startsWith('Writing') ? 'Writing the direction' : 'Generating what happens next'} title={phase.replace(/^[^“]*/, '').replace(/[“”]/g, '') || undefined}/></div> : <div className="replay-badge live-badge"><LoaderCircle className="spin" size={12}/> {phase}</div>)}{(() => { const canContinue = ready && live && !replay; return <form className={canContinue ? 'next-bar show' : 'next-bar'} aria-hidden={!canContinue} onSubmit={e => { e.preventDefault(); void continueStory(); }}><div className="next-row"><Sparkles size={15}/><input ref={nextInput} value={nextPrompt} disabled={busy || !canContinue} onChange={e => setNextPrompt(e.target.value)} placeholder={!canContinue ? 'Generate the storyboard first, then tell it what happens next…' : nextMode === 'continue' ? 'What happens next? The next clip starts from this exact frame…' : 'Describe the next shot. The model cuts to it with the same cast and location…'}/><div className="next-mode" role="radiogroup" aria-label="How to continue"><button type="button" className={nextMode === 'continue' ? 'active' : ''} aria-pressed={nextMode === 'continue'} disabled={busy || !canContinue} onClick={() => setNextMode('continue')}>Continue</button><button type="button" className={nextMode === 'cut' ? 'active' : ''} aria-pressed={nextMode === 'cut'} disabled={busy || !canContinue} onClick={() => setNextMode('cut')}>Cut</button></div><button className="primary" type="submit" disabled={busy || !canContinue || !nextPrompt.trim()}>{busy ? <LoaderCircle className="spin" size={14}/> : <Play size={14}/>} Play it</button></div><div className="next-chips">{['She hears a faint reply through the static and slowly turns toward the dunes.', 'Lights rise beyond the far dunes and she starts walking toward them.', 'She lowers the receiver and whispers: I am coming.'].map(c => <button key={c} type="button" disabled={busy || !canContinue} onClick={() => setNextPrompt(c)}>{c}</button>)}</div></form>; })()}</div><div className="transport"><div><span className={`status-dot ${ready ? 'online' : ''}`}/><span>{ready ? (preloading ? 'Model connected · uploading pictures' : picturesReady ? `Model connected · ${picturesReady} pictures ready` : 'Model connected') : status === 'disconnected' ? 'Model offline' : status}</span></div><div className="transport-actions"><button onClick={() => { setSound(!sound); if (audio.current) { audio.current.muted = sound; void audio.current.play().catch(() => {}); } }}><Volume2 size={15}/>{sound ? 'Mute' : 'Enable sound'}</button><button disabled={busy || !!playing || (!recordings[shot?.id] && take?.status !== 'ready')} onClick={() => void playShot()}><Play size={15}/> {recordings[shot?.id] ? 'Replay shot' : 'Play shot'}</button><button disabled={!recordings[shot?.id]} aria-label="Download take" onClick={downloadTake}><Download size={14}/></button><button disabled={!playing && !replay} aria-label="Stop playback" onClick={stopPlayback}><Square size={13}/></button>{present && <><button disabled={busy || !!playing || typeof replayPlan() === 'string'} onClick={playSequence}><Play size={15}/> Play sequence</button><button className="primary" disabled={!ready || busy || !board.shots.length} onClick={() => void generate(true)}>{busy ? <LoaderCircle className="spin" size={15}/> : <Clapperboard size={15}/>} Generate storyboard</button></>}<button onClick={togglePresent} aria-label={present ? 'Exit presentation' : 'Present'}>{present ? <Minimize2 size={14}/> : <Maximize2 size={14}/>} {present ? 'Exit' : 'Present'}</button></div><span>{shot ? durationLabel(shot.seconds) : ''}{shotBuild !== undefined ? ` · built in ${fmt(shotBuild)}` : ''} · 24 fps</span></div>
    <div className="sequence-heading"><div><h2>Sequence 01</h2><p>{timelineShots.length} shots <span className="monitor-separator">/</span> {fmt(filmSeconds)}</p></div><span className="speed-stat">{builtShots.length ? <>{fmt(builtShots.reduce((n, s) => n + s.seconds, 0))} of film generated in <strong>{fmt(buildTotal)}</strong>{builtShots.length < timelineShots.length ? ` · ${builtShots.length} of ${timelineShots.length} shots` : ''}</> : 'Faster than realtime · video and sound from one prompt'}</span></div><div className="timeline-ruler" aria-hidden="true">{[0, 1, 2, 3, 4].map(n => <span key={n}>{Math.floor(filmSeconds * n / 4).toString().padStart(2, '0')}:00</span>)}</div><div className="timeline-track"><span className="track-label" aria-hidden="true">V1</span><div className="filmstrip">{timelineShots.map((s, i) => <button key={s.id} className={`filmstrip-shot ${s.id === shot?.id ? 'selected' : ''} ${recordings[s.id] ? 'has-take' : ''}`} onClick={() => setSelected(s.id)}><div>{coverFor(s) ? <img src={coverFor(s)} alt=""/> : <Film/>}<span>{String(i + 1).padStart(2, '0')}</span><small>{durationLabel(s.seconds)}</small></div><strong>{s.title}</strong></button>)}</div></div>
    <div className={showLog ? 'log-panel open' : 'log-panel'}><button className="log-toggle" onClick={() => setShowLog(v => !v)}><Terminal size={12}/> Session log <span>{log.length}</span></button>{showLog && <><button className="log-clear" onClick={() => setLog([])}>Clear</button><ol className="log-lines">{[...log].reverse().map((l, i) => <li key={log.length - i} className={l.kind}><time>{new Date(l.t).toLocaleTimeString([], { hour12: false })}</time>{l.text}</li>)}{!log.length && <li className="empty">Connect the model to start logging.</li>}</ol></>}</div>
    </section>
    <aside className="inspector">{shot && <><div className="inspector-heading"><h2>Shot {String(index + 1).padStart(2, '0')}</h2><div><button aria-label="Delete shot" disabled={board.shots.length < 2 || busy || !!playing} onClick={() => { setBoard(b => ({ ...b, shots: b.shots.filter(s => s.id !== shot.id) })); setSelected(board.shots[index === 0 ? 1 : 0].id); }}><Trash2 size={15}/></button></div></div><div className="inspector-body"><label className="field">Shot name<input value={shot.title} onChange={e => update({ title: e.target.value })}/></label><label className="field prompt-field">Direction<textarea value={shot.prompt} onChange={e => update({ prompt: e.target.value })} placeholder="Describe the subject, camera move, light, any spoken line, and the sound…"/></label><p className="hint">Camera, light, dialogue, and sound all come from this one description. Name pictures as &lt;Picture 1&gt;, &lt;Picture 2&gt;…</p><label className="field">Length<select value={shot.seconds} onChange={e => update({ seconds: Number(e.target.value) })}>{durations.map(d => <option key={d} value={d}>{Math.round(d)} seconds</option>)}</select></label><div className="reference-heading"><h3>Pictures in this shot <span>{shot.refs.length}</span></h3><button onClick={() => setTab('references')} aria-label="Add a picture"><Plus size={15}/></button></div>{storyAssets.length ? <div className="reference-picker">{storyAssets.map(a => <button key={a.id} className={shot.refs.includes(a.id) ? 'chosen' : ''} aria-pressed={shot.refs.includes(a.id)} onClick={() => update({ refs: shot.refs.includes(a.id) ? shot.refs.filter(id => id !== a.id) : [...shot.refs, a.id] })}><img src={a.url} alt=""/><span>{a.name}</span>{shot.refs.includes(a.id) && <small>Picture {shot.refs.indexOf(a.id) + 1}</small>}</button>)}</div> : <button className="empty-references" onClick={() => setTab('references')}><ImagePlus size={21}/><span>Add a picture<small>An actor, a location, or a prop</small></span></button>}<label className="continuation"><input type="checkbox" checked={shot.connected} disabled={index === 0} onChange={e => update({ connected: e.target.checked })}/><span>Continue previous shot<small>Carry motion, camera, and sound forward</small></span></label></div><div className="generate-panel"><button className="primary generate" disabled={!ready || busy || !!playing || !shot.prompt.trim() || take?.status === 'queued'} onClick={() => void generate()}>{busy ? <LoaderCircle className="spin" size={17}/> : <Clapperboard size={17}/>} {recordings[shot.id] || take ? 'New take' : 'Generate shot'}</button><p>{busy ? 'Generating and saving a replayable take.' : recordings[shot.id] ? 'Saved on this device. Replay it with or without the model.' : !ready ? 'Connect the model to start generating.' : 'Every take is saved for sequence playback.'}</p></div></>}</aside></main>
  </div>;
}
