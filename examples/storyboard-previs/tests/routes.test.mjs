import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
registerHooks({ resolve(specifier, context, next) {
  if (specifier.startsWith('@/')) return next(new URL(`../${specifier.slice(2)}.ts`, import.meta.url).href, context);
  return next(specifier, context);
} });
const observe = await import('../app/api/observe/route.ts');
const direct = await import('../app/api/direct/route.ts');
// Synthetic values only. No upstream call is allowed in these tests.
delete process.env.OPENAI_API_KEY; delete process.env.ANTHROPIC_API_KEY;
globalThis.fetch = async () => { throw new Error('Unexpected upstream call'); };
const body = { source: 'scene', image: 'data:image/jpeg;base64,/9j/AA==', questions: [{ id: 'cue', condition: 'A radio is visible.' }] };
const req = (data, path = 'observe', headers = {}) => new Request(`http://localhost/api/${path}`, { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(data) });

test('monitor endpoint reports absent credentials without exposing any secrets', async () => {
  const data = await (await observe.GET()).json();
  assert.deepEqual(data, { provider: 'openai', model: 'gpt-6-luna', configured: false, director: false });
  assert.equal((await observe.POST(req(body))).status, 503);
});
test('monitor rejects cross-origin and malformed/oversized input before calling a provider', async () => {
  assert.equal((await observe.POST(req(body, 'observe', { origin: 'https://other.example' }))).status, 403);
  assert.equal((await observe.POST(req(body, 'observe', { 'sec-fetch-site': 'cross-site' }))).status, 403);
  assert.equal((await observe.POST(req({ ...body, questions: [] }))).status, 400);
  assert.equal((await observe.POST(req({ padding: 'x'.repeat(1700001) }))).status, 413);
});
test('monitoring calls only OpenAI Decisions with server-side credentials and normalizes answers', async () => {
  process.env.OPENAI_API_KEY = 'fake-openai';
  let calls = 0;
  globalThis.fetch = async (url, init) => {
    calls++;
    assert.equal(url, 'https://api.openai.com/v1/decisions');
    assert.deepEqual(init.headers, { Authorization: 'Bearer fake-openai', 'Content-Type': 'application/json' });
    const payload = JSON.parse(init.body);
    assert.equal(payload.model, 'gpt-6-luna');
    assert.deepEqual(payload.questions, [{ name: 'cue', type: 'predicate', instructions: body.questions[0].condition }]);
    assert.equal(payload.input[0].content.find(c => c.type === 'input_image').image_url, body.image);
    return Response.json({ answers: [{ type: 'predicate', name: 'cue', probability: .96 }] });
  };
  const response = await observe.POST(req(body));
  const data = await response.json();
  assert.equal(calls, 1);
  assert.equal(response.status, 200); assert.deepEqual(data.answers, [{ id: 'cue', probability: .96 }]);
  assert.equal(data.provider, 'openai'); assert.equal(JSON.stringify(data).includes('fake-'), false);
});
test('upstream errors are sanitized; invalid responses never produce a positive answer', async () => {
  globalThis.fetch = async () => new Response('private upstream details', { status: 429 });
  const limited = await observe.POST(req(body)); assert.equal(limited.status, 429); assert.doesNotMatch(await limited.text(), /private upstream details/);
  globalThis.fetch = async () => Response.json({ answers: [{ type: 'refusal', name: 'cue', probability: 1 }] });
  assert.deepEqual((await (await observe.POST(req(body))).json()).answers, [{ id: 'cue', probability: null }]);
  globalThis.fetch = async () => { throw new Error('sensitive provider error'); };
  const failed = await observe.POST(req(body)); assert.equal(failed.status, 502); assert.doesNotMatch(await failed.text(), /sensitive provider error/);
});
const direction = { beat: 'Make the next shot rainy.', mode: 'cut', story: { title: 'Test', shots: [] }, pictures: [], brief: 'Keep a hopeful ending.', observations: ['A person raised their hand.'] };
test('director validates input, checks origin, and requires its own credential', async () => {
  assert.equal((await direct.POST(req(direction, 'direct', { origin: 'https://other.example' }))).status, 403);
  assert.equal((await direct.POST(req({ ...direction, beat: {} }, 'direct'))).status, 400);
  assert.equal((await direct.POST(req({ padding: 'x'.repeat(100001) }, 'direct'))).status, 413);
  assert.equal((await direct.POST(req(direction, 'direct'))).status, 503);
});
test('director forwards structured brief and observations and rejects malformed output', async () => {
  process.env.ANTHROPIC_API_KEY = 'fake-director';
  globalThis.fetch = async (url, init) => {
    assert.equal(url, 'https://api.anthropic.com/v1/messages'); assert.equal(init.headers['x-api-key'], 'fake-director');
    const message = JSON.parse(JSON.parse(init.body).messages[0].content);
    assert.equal(message.creativeBrief, direction.brief); assert.deepEqual(message.observations, direction.observations);
    return Response.json({ content: [{ type: 'text', text: 'In one shot, rain falls.\nCamera: Wide.\nDialogue: None.\nSoundscape: Rain.\nMusic: None.' }] });
  };
  const response = await direct.POST(req(direction, 'direct')); assert.equal(response.status, 200); assert.match((await response.json()).prompt, /^In one shot,/);
  globalThis.fetch = async () => Response.json({ content: [{ type: 'text', text: 'I cannot produce direction.' }] });
  assert.equal((await direct.POST(req(direction, 'direct'))).status, 502);
});
