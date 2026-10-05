// Binary PNGs are sent in small ordered batches. The backend still spools and
// encodes one frame at a time, so complete episodes never form one large blob.
export async function createEpisodeExport(baseUrl, format = 'gif', fetchRequest = fetch) {
  const exportFormat = String(format || 'gif').toLowerCase();
  if (!['gif', 'mp4'].includes(exportFormat)) throw new Error('Episode export format must be GIF or MP4');
  const formatLabel = exportFormat.toUpperCase();

  async function request(path, method = 'POST', payload) {
    const response = await fetchRequest(`${baseUrl}/api/episode_exports${path}`, {
      method,
      ...(payload === undefined ? {} : {
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      if (response.status === 404 && (!path || path.startsWith('?'))) {
        throw new Error('Restart the backend to enable streaming episode exports');
      }
      throw new Error(error.detail || `${formatLabel} export request failed (${response.status})`);
    }
    return response.json();
  }

  const { export_id: id } = await request(`?format=${encodeURIComponent(exportFormat)}`);
  if (!id) throw new Error(`Backend did not create an ${formatLabel} export`);
  const path = `/${encodeURIComponent(id)}`;
  let frameCount = 0;
  let closed = false;
  return {
    async append(frame, duration) {
      if (closed) throw new Error(`${formatLabel} export is already closed`);
      if (!frame?.startsWith('data:image/png')) throw new Error('Board renderer returned an invalid PNG');
      const response = await request(`${path}/frames`, 'POST', { index: frameCount, frame, duration });
      if (response.frame_count !== frameCount + 1) throw new Error('Backend did not acknowledge the episode frame');
      frameCount += 1;
    },
    async appendBatch(frames) {
      if (closed) throw new Error(`${formatLabel} export is already closed`);
      if (!Array.isArray(frames) || frames.length === 0) return;
      const expectedCount = frameCount + frames.length;
      const metadata = frames.map((frame, offset) => {
        if (frame?.index !== frameCount + offset) throw new Error('GIF frame batch is out of order');
        if (!(frame?.png instanceof Blob) || frame.png.size === 0) {
          throw new Error('Board renderer returned an invalid PNG');
        }
        return { index: frame.index, duration: frame.duration, length: frame.png.size };
      });
      const metadataBytes = new TextEncoder().encode(JSON.stringify(metadata));
      const prefix = new Uint8Array(4);
      new DataView(prefix.buffer).setUint32(0, metadataBytes.byteLength);
      const response = await fetchRequest(`${baseUrl}/api/episode_exports${path}/frame_batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: new Blob([prefix, metadataBytes, ...frames.map((frame) => frame.png)]),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `${formatLabel} export request failed (${response.status})`);
      }
      const result = await response.json();
      if (result.frame_count !== expectedCount) throw new Error('Backend did not acknowledge the episode frame batch');
      frameCount = expectedCount;
    },
    async finish() {
      if (closed) throw new Error(`${formatLabel} export is already closed`);
      const response = await request(`${path}/finish`, 'POST', { frame_count: frameCount });
      closed = true;
      return response;
    },
    async abort() {
      if (closed) return;
      closed = true;
      await request(path, 'DELETE');
    },
  };
}

export function createEpisodeGifExport(baseUrl, fetchRequest = fetch) {
  return createEpisodeExport(baseUrl, 'gif', fetchRequest);
}
