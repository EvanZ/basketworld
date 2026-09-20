import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// Load this Vite ES module in Node without changing the app's package mode.
const source = await readFile(new URL('../src/utils/episodeExport.js', import.meta.url), 'utf8');
const { createEpisodeGifExport } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const ok = (body) => ({ ok: true, json: async () => body });

test('uploads >512 MiB cumulatively without creating a giant string or concurrent uploads', async () => {
  let count = 0;
  let bytes = 0;
  let largest = 0;
  let inFlight = 0;
  const writer = await createEpisodeGifExport('', async (url, options) => {
    assert.equal(inFlight++, 0);
    try {
      if (url.endsWith('/episode_exports')) return ok({ export_id: 'test' });
      const payload = JSON.parse(options.body);
      if (url.endsWith('/frames')) {
        assert.equal(payload.index, count++);
        largest = Math.max(largest, options.body.length);
        bytes += options.body.length;
        return ok({ frame_count: count });
      }
      assert.equal(payload.frame_count, 600);
      return ok({ status: 'success', frame_count: count });
    } finally { inFlight -= 1; }
  });
  const frame = 'data:image/png;base64,' + 'a'.repeat(1024 * 1024);
  for (let index = 0; index < 600; index += 1) await writer.append(frame, 0.1);
  assert.equal((await writer.finish()).frame_count, 600);
  assert.ok(bytes > 536870888);
  assert.ok(largest < 1.01 * 1024 * 1024);
  await writer.abort(); // Successful exports require no cancellation request.
});

test('upload failures surface and unfinished exports can be cancelled', async () => {
  const calls = [];
  const writer = await createEpisodeGifExport('', async (url, options) => {
    calls.push(options.method);
    if (calls.length === 1) return ok({ export_id: 'test' });
    if (options.method === 'DELETE') return ok({ status: 'cancelled' });
    return { ok: false, status: 400, json: async () => ({ detail: 'bad frame' }) };
  });
  await assert.rejects(writer.append('data:image/png;base64,AAAA', 1), /bad frame/);
  await writer.abort();
  assert.deepEqual(calls, ['POST', 'POST', 'DELETE']);
});

test('old backend gives an actionable restart message before capture', async () => {
  await assert.rejects(createEpisodeGifExport('', async () => ({
    ok: false, status: 404, json: async () => ({ detail: 'Not Found' }),
  })), /Restart the backend/);
});
