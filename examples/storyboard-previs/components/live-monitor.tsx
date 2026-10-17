'use client';

import { useEffect, useRef, useState } from 'react';
import { Activity, Camera, Code, Plus, Radio, RotateCcw, Square, Trash2, Volume2 } from 'lucide-react';
import { feedText, observeRule, parseRules, RULE_LIMIT, type DecisionAnswer, type LiveRule, type Observation, type RuleEvent, type RuleState, type WatchSource } from '@/lib/live-rules';

type Props = {
  enabled: boolean;
  onEnabled: (enabled: boolean) => void;
  getScene: () => Observation | null;
  sceneKey: string;
  shotPrompt: string;
  onEvent: (event: RuleEvent) => void;
  pendingDirection: string;
  onClearDirection: () => void;
  onApplyDirection: () => void;
  canApply: boolean;
  brief: string;
  onBrief: (value: string) => void;
};
const emptyState = (): RuleState => ({ hits: 0, fired: false });
const sourceName = { scene: 'Scene', webcam: 'Webcam', feed: 'Text feed' };
const actionName = { mark: 'Mark this moment', chime: 'Sound a cue', stop: 'Stop playback', direct: 'Direct the next shot' };
const defaultRules: LiveRule[] = [{ id: 'receiver-cue', source: 'scene', condition: 'A person is holding the shortwave radio receiver in their hands.', action: 'chime', direction: '', enabled: true }];

/** Draw the actual video pixels, never the UI or its reference-image placeholder. */
export function captureFrame(video: HTMLVideoElement | null): string | null {
  if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight || video.ended || video.paused) return null;
  const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 360;
  const context = canvas.getContext('2d'); if (!context) return null;
  context.fillStyle = '#000'; context.fillRect(0, 0, 640, 360);
  const scale = Math.min(640 / video.videoWidth, 360 / video.videoHeight);
  const w = video.videoWidth * scale, h = video.videoHeight * scale;
  context.drawImage(video, (640 - w) / 2, (360 - h) / 2, w, h);
  return canvas.toDataURL('image/jpeg', 0.7);
}

export function LiveMonitor(props: Props) {
  const [rules, setRules] = useState<LiveRule[]>(defaultRules);
  const [loaded, setLoaded] = useState(false);
  const [states, setStates] = useState<Record<string, RuleState>>({});
  const [events, setEvents] = useState<RuleEvent[]>([]);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('Monitoring is off');
  const [config, setConfig] = useState<{ provider: string; configured: boolean; director: boolean } | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [intervalMs, setIntervalMs] = useState(1000);
  const [judge, setJudge] = useState(false);
  const [autoAdjust, setAutoAdjust] = useState(false);
  const [qualityGoal, setQualityGoal] = useState('The main subject remains clearly visible.');
  const [quality, setQuality] = useState<number | null>(null);
  const [text, setText] = useState('');
  const [feedUrl, setFeedUrl] = useState('');
  const [pollUrl, setPollUrl] = useState('');
  const [feedError, setFeedError] = useState('');
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraBusy, setCameraBusy] = useState(false);
  const [code, setCode] = useState<string | null>(null);
  const webcam = useRef<HTMLVideoElement>(null);
  const camera = useRef<MediaStream | null>(null);
  const cameraGeneration = useRef(0);
  const audioContext = useRef<AudioContext | null>(null);
  const stateRef = useRef<Record<string, RuleState>>({});
  const latest = useRef({ props, rules, text, judge, autoAdjust, qualityGoal });
  latest.current = { props, rules, text, judge, autoAdjust, qualityGoal };
  const revision = useRef(0);
  const activeRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    try { const saved = localStorage.getItem('previs-live-rules-v1'); if (saved) setRules(parseRules(JSON.parse(saved))); }
    catch { setError('Saved rules could not be read. The example rule is loaded.'); }
    setLoaded(true);
    const controller = new AbortController();
    void fetch('/api/observe', { signal: controller.signal }).then(r => r.json()).then(data => {
      if (data.error) setError(data.error); else setConfig(data);
    }).catch(e => { if (e.name !== 'AbortError') setError('Could not check monitoring configuration.'); });
    return () => { controller.abort(); cameraGeneration.current++; camera.current?.getTracks().forEach(t => t.stop()); void audioContext.current?.close(); };
  }, []);
  useEffect(() => {
    if (!loaded) return;
    try { parseRules(rules); } catch { return; } // Keep the last valid rules while a draft is incomplete.
    try { localStorage.setItem('previs-live-rules-v1', JSON.stringify(rules)); } catch { setError('Rules could not be saved on this device.'); }
  }, [rules, loaded]);
  useEffect(() => { if (!props.enabled) stopCamera(); }, [props.enabled]);
  useEffect(() => {
    // Edits, provider stop, and shot changes invalidate all work already in flight.
    revision.current++; activeRequest.current?.abort();
    stateRef.current = Object.fromEntries(Object.entries(stateRef.current).map(([id, state]) => [id, { ...state, hits: 0, lastKey: undefined }]));
    setStates({ ...stateRef.current });
  }, [props.enabled, props.sceneKey, rules, judge, qualityGoal, autoAdjust, text, cameraOn]);

  useEffect(() => { delete stateRef.current['quality']; setQuality(null); }, [props.sceneKey, judge, qualityGoal]);

  function resetRule(id: string) {
    revision.current++; activeRequest.current?.abort();
    stateRef.current[id] = emptyState(); setStates({ ...stateRef.current });
  }
  function editRule(id: string, patch: Partial<LiveRule>) {
    resetRule(id); setRules(current => current.map(rule => rule.id === id ? { ...rule, ...patch } : rule));
  }
  function stopCamera() {
    cameraGeneration.current++;
    camera.current?.getTracks().forEach(t => t.stop()); camera.current = null;
    if (webcam.current) webcam.current.srcObject = null;
    setCameraOn(false); setCameraBusy(false);
  }
  async function startCamera() {
    const generation = ++cameraGeneration.current; setCameraBusy(true); setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 }, audio: false });
      if (generation !== cameraGeneration.current) { stream.getTracks().forEach(t => t.stop()); return; }
      camera.current = stream;
      if (webcam.current) { webcam.current.srcObject = stream; await webcam.current.play(); }
      setCameraOn(true);
    } catch (e) { if (generation === cameraGeneration.current) { stopCamera(); setError(`Webcam unavailable: ${e instanceof Error ? e.message : String(e)}`); } }
    finally { if (generation === cameraGeneration.current) setCameraBusy(false); }
  }
  function chime() {
    const context = audioContext.current;
    if (!context || context.state !== 'running') { setError('Cue audio is blocked. Stop and restart monitoring with a click.'); return; }
    const oscillator = context.createOscillator(), gain = context.createGain();
    oscillator.frequency.value = 740; gain.gain.setValueAtTime(0.12, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.3);
    oscillator.connect(gain); gain.connect(context.destination); oscillator.start(); oscillator.stop(context.currentTime + 0.3);
  }
  function publish(event: RuleEvent) {
    setEvents(history => [...history.slice(-29), event]);
    if (event.action === 'chime') chime();
    if (event.action === 'stop') { revision.current++; activeRequest.current?.abort(); latest.current.props.onEnabled(false); }
    window.dispatchEvent(new CustomEvent('previs:trigger', { detail: event }));
    latest.current.props.onEvent(event);
  }

  useEffect(() => {
    if (!props.enabled) { setStatus('Monitoring is off'); return; }
    let disposed = false, timer: ReturnType<typeof setTimeout>, cursor = 0, failures = 0;
    let controller: AbortController | null = null;
    const previous: Partial<Record<WatchSource, Observation>> = {};
    async function tick() {
      const started = performance.now();
      try {
        if (document.hidden) { setStatus('Waiting for this tab'); return; }
        const current = latest.current;
        const sources = (['scene', 'webcam', 'feed'] as const).filter(source => current.rules.some(r => r.enabled && r.source === source && !stateRef.current[r.id]?.fired) || (source === 'scene' && current.judge));
        if (!sources.length) { setStatus('All rules fired. Rearm a rule to watch again.'); return; }
        const source = sources[cursor++ % sources.length];
        let observation: Observation | null = null;
        if (source === 'scene') observation = current.props.getScene();
        if (source === 'webcam' && camera.current) {
          const frame = captureFrame(webcam.current);
          if (frame) observation = { source, image: frame, key: `webcam:${webcam.current!.currentTime}`, capturedAt: Date.now(), seconds: webcam.current!.currentTime, shotId: current.props.sceneKey };
        }
        if (source === 'feed' && current.text.trim()) observation = { source, text: current.text, key: `feed:${Date.now()}`, capturedAt: Date.now(), seconds: 0, shotId: current.props.sceneKey };
        if (!observation || (source !== 'feed' && observation.key === previous[source]?.key)) { setStatus(`Waiting for ${sourceName[source].toLowerCase()} input`); return; }
        const watched = current.rules.filter(r => r.enabled && r.condition.trim() && r.source === source && !stateRef.current[r.id]?.fired && (r.action !== 'direct' || r.direction.trim()));
        const questions = watched.map(r => ({ id: r.id, condition: r.condition }));
        if (source === 'scene' && current.judge && current.qualityGoal.trim()) questions.push({ id: 'quality', condition: `Does the current frame clearly VIOLATE this visual requirement: ${current.qualityGoal}? Judge only what can be seen, not future actions or audio. The intended shot is context, not proof: ${current.props.shotPrompt.slice(0, 2500)}` });
        if (!questions.length) return;
        const capturedRevision = revision.current;
        controller = new AbortController(); activeRequest.current = controller;
        const before = previous[source]; previous[source] = observation;
        setStatus(`Watching ${sourceName[source].toLowerCase()}`);
        const response = await fetch('/api/observe', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source, image: observation.image, text: observation.text, previousImage: before?.shotId === observation.shotId ? before.image : undefined, questions }),
          signal: controller.signal,
        });
        const result = await response.json() as { answers: DecisionAnswer[]; latencyMs: number; error?: string };
        if (disposed || capturedRevision !== revision.current) return;
        if (!response.ok) throw new Error(result.error || 'Monitoring failed.');
        if (source === 'scene' && latest.current.props.getScene()?.shotId !== observation.shotId) return;
        failures = 0; setError(''); setLatency(result.latencyMs);
        for (const answer of result.answers) {
          if (disposed || capturedRevision !== revision.current) break;
          const prior = stateRef.current[answer.id] || emptyState();
          const next = observeRule(prior, observation.key, answer.probability);
          stateRef.current[answer.id] = next.state;
          if (answer.id === 'quality') {
            setQuality(answer.probability);
            if (next.fire) publish({ ...observation, id: crypto.randomUUID(), ruleId: 'quality', condition: `Visual requirement may be violated: ${current.qualityGoal}`, probability: answer.probability!, action: current.autoAdjust ? 'direct' : 'mark', direction: `Correct this visual issue in the next shot: ${current.qualityGoal}. Preserve the story and cast.` });
          } else if (next.fire) {
            const rule = watched.find(r => r.id === answer.id);
            if (rule) publish({ ...observation, id: crypto.randomUUID(), ruleId: rule.id, condition: rule.condition, probability: answer.probability!, action: rule.action, direction: rule.direction });
          }
        }
        setStates({ ...stateRef.current });
      } catch (e) {
        if (disposed || (e instanceof Error && e.name === 'AbortError')) return;
        failures++; setError(e instanceof Error ? e.message : 'Monitoring failed.');
        stateRef.current = Object.fromEntries(Object.entries(stateRef.current).map(([id, state]) => [id, { ...state, hits: 0 }]));
        if (failures >= 3) latest.current.props.onEnabled(false);
      } finally {
        if (activeRequest.current === controller) activeRequest.current = null;
        if (!disposed) timer = setTimeout(() => void tick(), Math.max(intervalMs - (performance.now() - started), failures ? 3000 : 0));
      }
    }
    void tick();
    return () => { disposed = true; clearTimeout(timer); controller?.abort(); };
  }, [props.enabled, intervalMs]);

  useEffect(() => {
    if (!props.enabled || !pollUrl) return;
    let stopped = false, timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    async function poll() {
      try {
        const response = await fetch(pollUrl, { mode: 'cors', credentials: 'omit', cache: 'no-store', signal: AbortSignal.any([controller.signal, AbortSignal.timeout(6000)]) });
        if (!response.ok) throw new Error(`Feed returned ${response.status}.`);
        const reader = response.body?.getReader(); if (!reader) throw new Error('Empty feed.');
        const chunks: Uint8Array[] = []; let length = 0;
        while (true) { const { done, value } = await reader.read(); if (done) break; length += value.length; if (length > 64000) { await reader.cancel(); throw new Error('Feed exceeds 64 KB.'); } chunks.push(value); }
        const bytes = new Uint8Array(length); let offset = 0; for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
        const next = feedText(JSON.parse(new TextDecoder().decode(bytes)));
        if (!stopped) { setText(next); setFeedError(''); }
      } catch (e) { if (!stopped) { setText(''); setFeedError(`Feed unavailable. Use a CORS-enabled JSON URL or paste text. ${e instanceof Error ? e.message : ''}`); } }
      finally { if (!stopped) timer = setTimeout(() => void poll(), 10000); }
    }
    void poll();
    return () => { stopped = true; controller.abort(); clearTimeout(timer); };
  }, [props.enabled, pollUrl]);

  function toggle() {
    if (!props.enabled) {
      setError('');
      try { parseRules(rules); } catch (e) { setError(String(e)); return; }
      if (!audioContext.current || audioContext.current.state === 'closed') audioContext.current = new AudioContext();
      void audioContext.current.resume().catch(() => setError('Audio cues are unavailable in this browser.'));
    } else stopCamera();
    props.onEnabled(!props.enabled);
  }

  return <section className="live-monitor" aria-label="Live rules">
    <div className="watch-heading"><h2><Activity size={15}/> Live rules</h2><button className={props.enabled ? 'watch-active' : 'primary'} disabled={!config?.configured} onClick={toggle}>{props.enabled ? <Square size={12}/> : <Radio size={12}/>} {props.enabled ? 'Stop monitoring' : 'Start monitoring'}</button></div>
    <p className="watch-status" role="status">{status}{latency !== null && props.enabled ? ` · ${latency} ms` : ''} <span>{config ? 'OpenAI Decisions' : 'Checking provider…'}</span></p>
    {config && !config.configured && <p className="watch-help">Set OPENAI_API_KEY on the server to enable monitoring. Rules can be edited now.</p>}
    {error && <p className="watch-error" role="alert">{error}</p>}
    {!!events.length && <div className="watch-last" aria-label="Latest trigger">{events.at(-1)?.image && <img src={events.at(-1)!.image} alt="Latest trigger evidence"/>}<div><small>Latest cue · {events.at(-1)!.seconds.toFixed(1)}s</small><p>{events.at(-1)!.condition}</p></div></div>}
    <div className="watch-settings"><label>Check every<select value={intervalMs} onChange={e => setIntervalMs(Number(e.target.value))}><option value={500}>0.5 seconds</option><option value={1000}>1 second</option><option value={2000}>2 seconds</option></select></label><button onClick={() => setCode(code === null ? JSON.stringify(rules, null, 2) : null)}><Code size={13}/> Rule JSON</button></div>
    <p className="watch-help">Watches playing video. Two checks at 85% or above fire a rule once. Multiple sources share the loop; provider latency can slow checks.</p>
    {code !== null && <div className="watch-code"><textarea aria-label="Rule JSON" value={code} onChange={e => setCode(e.target.value)} spellCheck={false}/><button onClick={() => { try { const parsed = parseRules(JSON.parse(code)); stateRef.current = {}; setStates({}); setRules(parsed); setCode(null); setError(''); } catch (e) { setError(e instanceof Error ? e.message : 'Invalid rules.'); } }}>Apply rules</button></div>}
    <div className="watch-rules">{rules.map(rule => <article className="watch-rule" key={rule.id}>
      <div className="watch-rule-top"><label><input type="checkbox" checked={rule.enabled} onChange={e => editRule(rule.id, { enabled: e.target.checked })}/> Armed</label><span className={states[rule.id]?.fired ? 'fired' : ''}>{states[rule.id]?.fired ? 'Fired' : states[rule.id]?.probability !== undefined ? `${Math.round(states[rule.id].probability! * 100)}%` : 'Waiting'}</span><button aria-label="Rearm rule" onClick={() => resetRule(rule.id)}><RotateCcw size={12}/></button><button aria-label="Delete rule" onClick={() => setRules(rules.filter(r => r.id !== rule.id))}><Trash2 size={12}/></button></div>
      <label>Watch<select value={rule.source} onChange={e => editRule(rule.id, { source: e.target.value as WatchSource })}>{Object.entries(sourceName).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <label>When<textarea rows={2} maxLength={500} value={rule.condition} onChange={e => editRule(rule.id, { condition: e.target.value })}/></label>
      <label>Then<select value={rule.action} onChange={e => editRule(rule.id, { action: e.target.value as LiveRule['action'] })}>{Object.entries(actionName).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      {rule.action === 'direct' && <label>Direction<textarea rows={2} maxLength={500} placeholder="She hears a reply and looks toward the antenna." value={rule.direction} onChange={e => editRule(rule.id, { direction: e.target.value })}/></label>}
    </article>)}</div>
    <button className="watch-add" disabled={rules.length >= RULE_LIMIT} onClick={() => setRules([...rules, { id: crypto.randomUUID(), source: 'scene', condition: '', action: 'mark', direction: '', enabled: true }])}><Plus size={13}/> Add rule</button>
    <details className="watch-section"><summary>Judge the output</summary><label className="watch-check"><input type="checkbox" checked={judge} onChange={e => setJudge(e.target.checked)}/> Check a visual requirement</label><textarea aria-label="Visual requirement" maxLength={500} rows={2} value={qualityGoal} onChange={e => setQualityGoal(e.target.value)}/><p className="watch-help">{quality === null ? 'Looks for clear violations while the scene plays.' : `Estimated probability of a violation: ${Math.round(quality * 100)}%.`}</p><label className="watch-check"><input type="checkbox" checked={autoAdjust} disabled={!judge || !config?.director} onChange={e => setAutoAdjust(e.target.checked)}/> Ask the director to correct the next shot</label></details>
    <details className="watch-section"><summary>External inputs</summary><p className="watch-help">Webcam frames go to OpenAI only while monitoring a webcam rule. Audio is never captured.</p><video ref={webcam} autoPlay playsInline muted className={cameraOn ? 'watch-camera visible' : 'watch-camera'}/><button onClick={() => cameraOn || cameraBusy ? stopCamera() : void startCamera()}><Camera size={13}/>{cameraBusy ? 'Cancel webcam' : cameraOn ? 'Turn webcam off' : 'Enable webcam'}</button><label>Text or social feed<textarea maxLength={4000} rows={3} placeholder="Paste audience comments, headlines or another text signal…" value={text} onChange={e => { setPollUrl(''); setText(e.target.value); }}/></label><label>JSON feed URL<input type="url" value={feedUrl} placeholder="https://…/feed.json" onChange={e => setFeedUrl(e.target.value)}/></label><div className="watch-settings"><button onClick={() => { try { const url = new URL(feedUrl); if (url.protocol !== 'https:' || url.username || url.password) throw new Error('Use an HTTPS URL without credentials.'); setPollUrl(url.href); setFeedError(''); } catch (e) { setFeedError(e instanceof Error ? e.message : 'Invalid URL.'); } }}>Connect feed</button><button disabled={!pollUrl} onClick={() => { setPollUrl(''); setText(''); }}>Disconnect feed</button></div><p className="watch-help">{pollUrl ? 'Feed connected; refreshes every 10 seconds while monitoring.' : 'Use a public CORS-enabled JSON feed: { "text": "…" }. Social API credentials belong in your own feed adapter.'}</p>{feedError && <p className="watch-error" role="alert">{feedError}</p>}</details>
    <details className="watch-section" open><summary>Director memory</summary><label>Creative brief<textarea rows={2} maxLength={1000} value={props.brief} onChange={e => props.onBrief(e.target.value)} placeholder="Build tension; keep the watcher hidden until the final shot."/></label><p className="watch-help">{config?.director ? 'Uses the brief and recent detections for upcoming shots. Start generation with monitoring on to adapt one shot at a time; already generated video cannot be changed.' : 'Set ANTHROPIC_API_KEY to enable director actions.'}</p>{props.pendingDirection && <div className="watch-pending"><strong>Next shot</strong><p>{props.pendingDirection}</p><div className="watch-settings"><button disabled={!props.canApply} onClick={props.onApplyDirection}>Generate next beat</button><button onClick={props.onClearDirection}>Cancel</button></div></div>}</details>
    <details className="watch-section" open><summary>Detected moments <span>{events.length}</span></summary>{!events.length && <p className="watch-help">Events appear here with their evidence. Each also emits a previs:trigger browser event.</p>}{[...events].reverse().map(event => <div className="watch-event" key={event.id}>{event.image && <img src={event.image} alt={`Evidence for ${event.condition}`}/>}<div><small>{sourceName[event.source]} · {event.seconds.toFixed(1)}s · {Math.round(event.probability * 100)}%</small><p>{event.condition}</p><span>{event.action === 'chime' && <Volume2 size={10}/>} {actionName[event.action]}</span></div></div>)}</details>
  </section>;
}
