import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// Load this Vite ES module in Node without changing the app's package mode.
const source = await readFile(new URL('../src/utils/episodeExport.js', import.meta.url), 'utf8');
const { createEpisodeExport, createEpisodeGifExport } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const ok = (body) => ({ ok: true, json: async () => body });

test('uploads >512 MiB cumulatively in bounded binary batches', async () => {
  let count = 0;
  let bytes = 0;
  let largest = 0;
  let inFlight = 0;
  const writer = await createEpisodeGifExport('', async (url, options) => {
    assert.equal(inFlight++, 0);
    try {
      if (url.endsWith('/episode_exports?format=gif')) return ok({ export_id: 'test' });
      if (url.endsWith('/frame_batch')) {
        assert.equal(options.headers['Content-Type'], 'application/octet-stream');
        assert.ok(options.body instanceof Blob);
        const data = new Uint8Array(await options.body.arrayBuffer());
        const metadataSize = new DataView(data.buffer, data.byteOffset, 4).getUint32(0);
        const metadata = JSON.parse(new TextDecoder().decode(data.slice(4, 4 + metadataSize)));
        assert.ok(metadata.length > 0 && metadata.length <= 8);
        assert.deepEqual(metadata.map((frame) => frame.index),
          Array.from({ length: metadata.length }, (_value, index) => count + index));
        const payloadLength = data.byteLength;
        largest = Math.max(largest, payloadLength);
        bytes += payloadLength;
        count += metadata.length;
        return ok({ frame_count: count });
      }
      const payload = JSON.parse(options.body);
      assert.equal(payload.frame_count, 600);
      return ok({ status: 'success', frame_count: count });
    } finally { inFlight -= 1; }
  });
  const png = new Blob([new Uint8Array(1024 * 1024)], { type: 'image/png' });
  let batch = [];
  for (let index = 0; index < 600; index += 1) {
    batch.push({ index, png, duration: 0.1 });
    if (batch.length === 8) {
      await writer.appendBatch(batch);
      batch = [];
    }
  }
  await writer.appendBatch(batch);
  assert.equal((await writer.finish()).frame_count, 600);
  assert.ok(bytes > 600 * 1024 * 1024);
  assert.ok(largest < 8.01 * 1024 * 1024);
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
  await assert.rejects(writer.appendBatch([
    { index: 0, png: new Blob(['png'], { type: 'image/png' }), duration: 1 },
  ]), /bad frame/);
  await writer.abort();
  assert.deepEqual(calls, ['POST', 'POST', 'DELETE']);
});

test('old backend gives an actionable restart message before capture', async () => {
  await assert.rejects(createEpisodeGifExport('', async () => ({
    ok: false, status: 404, json: async () => ({ detail: 'Not Found' }),
  })), /Restart the backend/);
});

test('MP4 export selects the movie backend and retains bounded batches', async () => {
  const urls = [];
  const writer = await createEpisodeExport('', 'mp4', async (url, options) => {
    urls.push(url);
    if (url.endsWith('/episode_exports?format=mp4')) {
      return ok({ export_id: 'movie', format: 'mp4' });
    }
    if (url.endsWith('/frame_batch')) return ok({ frame_count: 1 });
    if (url.endsWith('/finish')) {
      return ok({ status: 'success', frame_count: 1, file_path: '/tmp/episode.mp4' });
    }
    throw new Error(`Unexpected request ${url} ${options.method}`);
  });
  await writer.appendBatch([
    { index: 0, png: new Blob(['png'], { type: 'image/png' }), duration: 1 },
  ]);
  const result = await writer.finish();
  assert.equal(result.file_path, '/tmp/episode.mp4');
  assert.equal(urls[0], '/api/episode_exports?format=mp4');
});
