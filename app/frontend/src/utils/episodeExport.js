// One upload per frame: never JSON.stringify an entire animated episode.
// The backend spools PNGs to disk and encodes one frame at a time as well.
export async function createEpisodeGifExport(baseUrl, fetchRequest = fetch) {
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
      if (response.status === 404 && !path) {
        throw new Error('Restart the backend to enable streaming GIF exports');
      }
      throw new Error(error.detail || `GIF export request failed (${response.status})`);
    }
    return response.json();
  }

  const { export_id: id } = await request('');
  if (!id) throw new Error('Backend did not create a GIF export');
  const path = `/${encodeURIComponent(id)}`;
  let frameCount = 0;
  let closed = false;
  return {
    async append(frame, duration) {
      if (closed) throw new Error('GIF export is already closed');
      if (!frame?.startsWith('data:image/png')) throw new Error('Board renderer returned an invalid PNG');
      const response = await request(`${path}/frames`, 'POST', { index: frameCount, frame, duration });
      if (response.frame_count !== frameCount + 1) throw new Error('Backend did not acknowledge the GIF frame');
      frameCount += 1;
    },
    async finish() {
      if (closed) throw new Error('GIF export is already closed');
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
