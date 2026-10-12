const API_ROOT = import.meta.env?.VITE_TRAINING_API_URL || '/api/v1'

export function responseErrorMessage(payload, status) {
  const detail = payload?.detail
  if (typeof detail === 'string') return detail
  if (typeof detail?.message === 'string') return detail.message
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg)
      .filter((message) => typeof message === 'string' && message.trim())
      .map((message) => message.replace(/^Value error,\s*/i, ''))
    if (messages.length) return [...new Set(messages)].join(' ')
  }
  return `Request failed (${status})`
}

async function request(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    const detail = payload.detail
    const message = responseErrorMessage(payload, response.status)
    const error = new Error(message)
    error.detail = detail
    throw error
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
  getHistoricalElo: (id) => request(`/runs/${id}/elo`),
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
  resumeRun: (id, confirmProvenanceMismatch = false) => request(`/runs/${id}/resume`, {
    method: 'POST',
    body: JSON.stringify({ confirm_provenance_mismatch: confirmProvenanceMismatch }),
  }),
  continueRun: (id, additionalUpdates, confirmProvenanceMismatch = false) => request(`/runs/${id}/continue`, {
    method: 'POST',
    body: JSON.stringify({
      additional_updates: additionalUpdates,
      confirm_provenance_mismatch: confirmProvenanceMismatch,
    }),
  }),
  action: (id, action) => request(`/runs/${id}/${action}`, { method: 'POST' }),
}
