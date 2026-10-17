import test from 'node:test';
import assert from 'node:assert/strict';
import { isAllowedOrigin } from '../lib/request-origin.ts';
const external = 'https://previs.example.ts.net';
const request = headers => new Request('http://127.0.0.1:3006/api/observe', { headers });
test('allows the configured HTTPS proxy origin and local requests', () => {
  assert.equal(isAllowedOrigin(request({ origin: external }), external), true);
  assert.equal(isAllowedOrigin(request({ origin: external }), external + '/'), true);
  assert.equal(isAllowedOrigin(request({ origin: 'http://127.0.0.1:3006' })), true);
  assert.equal(isAllowedOrigin(request({})), true);
});
test('does not trust arbitrary origins or spoofed forwarding headers', () => {
  assert.equal(isAllowedOrigin(request({ origin: external })), false);
  assert.equal(isAllowedOrigin(request({ origin: 'https://attacker.example', 'x-forwarded-host': 'attacker.example', 'x-forwarded-proto': 'https' }), external), false);
  assert.equal(isAllowedOrigin(request({ origin: external, 'sec-fetch-site': 'cross-site' }), external), false);
  for (const invalid of ['*', 'http://previs.example.ts.net', external + '/path', external + '?query', external + '#fragment', 'https://user:pass@previs.example.ts.net']) assert.equal(isAllowedOrigin(request({ origin: external }), invalid), false);
});
