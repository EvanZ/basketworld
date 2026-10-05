const API_ROOT = import.meta.env.VITE_TRAINING_API_URL || '/api/v1'

async function request(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

async function requestText(path) {
  const response = await fetch(`${API_ROOT}${path}`)
  if (!response.ok) throw new Error(`Request failed (${response.status})`)
  return response.text()
}

export const api = {
  getConfigSchema: () => request('/config-schema'),
  importMlflowRun: (trackingUri, runId) => request('/mlflow/import-run', {
    method: 'POST',
    body: JSON.stringify({ tracking_uri: trackingUri, run_id: runId }),
  }),
  listRuns: () => request('/runs'),
  getRun: (id) => request(`/runs/${id}`),
  getLogs: (id) => requestText(`/runs/${id}/logs`),
  getMetricCatalog: (id) => request(`/runs/${id}/metrics/catalog`),
  getMetricHistories: (id, names) => {
    const params = new URLSearchParams()
    names.forEach((name) => params.append('names', name))
    params.set('max_points', '240')
    return request(`/runs/${id}/metrics?${params.toString()}`)
  },
  previewRun: (config) => request('/runs/preview', {
    method: 'POST',
    body: JSON.stringify(config),
  }),
  createRun: (config, idempotencyKey) => request('/runs', {
    method: 'POST',
    body: JSON.stringify({ config, idempotency_key: idempotencyKey }),
  }),
  action: (id, action) => request(`/runs/${id}/${action}`, { method: 'POST' }),
}
