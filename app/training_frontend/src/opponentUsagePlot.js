function axis(rawMaximum, cap = Infinity) {
  const magnitude = 10 ** Math.floor(Math.log10(rawMaximum / 4))
  const ratio = rawMaximum / (4 * magnitude)
  const step = (ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10) * magnitude
  const maximum = Math.min(cap, Math.ceil(rawMaximum / step) * step)
  return { maximum, ticks: Array.from({ length: Math.round(maximum / step) + 1 }, (_, i) => i * step) }
}

export function opponentUsagePlot(rows, width) {
  const left = 62, right = Math.max(280, width) - 64, bottom = 202, top = 18
  const first = rows[0]?.update ?? 0, last = rows.at(-1)?.update ?? 1
  const span = Math.max(1, last - first)
  const gap = rows.length > 1 ? Math.min(...rows.slice(1).map((row, i) => Math.max(1, row.update - rows[i].update))) : span
  const lo = first - gap * 0.6, hi = last + gap * 0.6
  const x = value => left + (value - lo) / (hi - lo) * (right - left)
  const usageAxis = axis(Math.min(100, Math.max(0.1, ...rows.map(row => row.usage * 1.15))), 100)
  const y = value => bottom - value / usageAxis.maximum * (bottom - top)
  const pppValues = rows.flatMap(row => [row.pppScored, row.pppAllowed]).filter(Number.isFinite)
  const pppAxis = axis(Math.max(1, ...pppValues.map(value => value * 1.1)))
  const pppY = value => bottom - value / pppAxis.maximum * (bottom - top)
  const barWidth = Math.max(1, Math.min(44, gap / (hi - lo) * (right - left) * 0.7))
  const tickCount = width < 450 ? 3 : 5
  const xTicks = [...new Set(Array.from({ length: tickCount }, (_, i) => rows[Math.round((rows.length - 1) * i / (tickCount - 1))]?.update ?? 0))]
  const groups = new Map()
  for (const row of rows) groups.set(row.update, [...(groups.get(row.update) || []), row.opponent])
  const bars = rows.map(row => {
    const group = groups.get(row.update)
    return { ...row, x: x(row.update) + ((group.indexOf(row.opponent) + 0.5) / group.length - 0.5) * barWidth,
      y: y(row.usage), width: Math.max(1, barWidth / group.length * (group.length > 1 ? 0.9 : 1)) }
  })
  const series = [['pppScored', 'PPP scored', 'scored'], ['pppAllowed', 'PPP allowed', 'allowed']].map(([key, label, style]) => {
    let connected = false
    const path = bars.map(bar => {
      if (!Number.isFinite(bar[key])) { connected = false; return '' }
      const command = connected ? 'L' : 'M'
      connected = true
      return `${command}${bar.x},${pppY(bar[key])}`
    }).join(' ')
    const points = bars.filter(bar => Number.isFinite(bar[key])).map(bar => ({ ...bar, value: bar[key], y: pppY(bar[key]) }))
    return { key, label, style, path, points }
  })
  return { left, right, bottom, top, x, y, xTicks, uniqueUpdates: groups.size === rows.length,
    yTicks: usageAxis.ticks, pppY, pppTicks: pppValues.length ? pppAxis.ticks : [], bars, series }
}
