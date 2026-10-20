import { useMemo, useState } from 'react'

const SAMPLE_FIELDS = {
  ttft_ms: 'ttft_ms_samples',
  tbt_ms_p50: 'tbt_ms_samples',
  tbt_ms_mean: 'tbt_ms_samples',
  kv_bytes: 'kv_bytes_samples',
}

function percentile(sorted, p) {
  if (!sorted.length) return Number.NaN
  const i = (sorted.length - 1) * p
  const lo = Math.floor(i)
  const hi = Math.ceil(i)
  if (lo === hi) return sorted[lo]
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
}

function boxFive(values) {
  const s = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b)
  if (!s.length) return null
  const q1 = percentile(s, 0.25)
  const med = percentile(s, 0.5)
  const q3 = percentile(s, 0.75)
  const iqr = q3 - q1
  const loFence = q1 - 1.5 * iqr
  const hiFence = q3 + 1.5 * iqr
  // Tukey whiskers: end at the most extreme real sample inside the fences.
  const inside = s.filter((v) => v >= loFence && v <= hiFence)
  const lo = inside.length ? inside[0] : q1
  const hi = inside.length ? inside[inside.length - 1] : q3
  const outliers = s.filter((v) => v < loFence || v > hiFence)
  return { lo, q1, med, q3, hi, outliers, n: s.length, max: s[s.length - 1], min: s[0] }
}

function scaleY(v, yMin, yMax, h, padT) {
  if (yMax <= yMin) return padT + h / 2
  return padT + h - ((v - yMin) / (yMax - yMin)) * h
}

function BoxGlyph({ stats, cx, boxW, color, yMin, yMax, plotH, padT }) {
  const y = (v) => scaleY(v, yMin, yMax, plotH, padT)
  const x0 = cx - boxW / 2
  const medY = y(stats.med)
  const q1Y = y(stats.q1)
  const q3Y = y(stats.q3)
  const loY = y(stats.lo)
  const hiY = y(stats.hi)
  return (
    <g>
      <line x1={cx} y1={loY} x2={cx} y2={hiY} stroke={color} strokeWidth={1.5} />
      <line x1={x0} y1={loY} x2={cx + boxW / 2} y2={loY} stroke={color} strokeWidth={1.5} />
      <line x1={x0} y1={hiY} x2={cx + boxW / 2} y2={hiY} stroke={color} strokeWidth={1.5} />
      <rect
        x={x0}
        y={q3Y}
        width={boxW}
        height={Math.max(q1Y - q3Y, 1)}
        fill={color}
        fillOpacity={0.35}
        stroke={color}
        strokeWidth={1.5}
      />
      <line x1={x0} y1={medY} x2={x0 + boxW} y2={medY} stroke={color} strokeWidth={2} />
      {stats.outliers.map((v, i) => (
        <circle
          key={`out-${i}`}
          cx={cx}
          cy={y(v)}
          r={3.5}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
        >
          <title>{v}</title>
        </circle>
      ))}
    </g>
  )
}

function runMatchesFilter(run, filter) {
  if (!filter) return true
  return Object.entries(filter).every(([key, val]) => run[key] === val)
}

function samplesForRun(run, samplesField, metricKey) {
  const arr = run[samplesField]
  if (Array.isArray(arr) && arr.length) {
    return arr.filter((v) => Number.isFinite(v))
  }
  const scalar = run[metricKey]
  if (Number.isFinite(scalar)) return [scalar]
  return []
}

function samplesFieldForMetric(metricKey, cfg) {
  if (SAMPLE_FIELDS[metricKey]) return SAMPLE_FIELDS[metricKey]
  if (cfg.metric === metricKey && cfg.samples_field) return cfg.samples_field
  return `${metricKey}_samples`
}

function boxplotTitle(metricLabel, activeFilter) {
  const parts = Object.entries(activeFilter ?? {}).map(([k, v]) => `${k} = ${v}`)
  const filterSuffix = parts.length ? ` (${parts.join(', ')})` : ''
  return `${metricLabel} distribution per client${filterSuffix}`
}

function baseFilterWithoutConcurrency(cfg) {
  const f = { ...(cfg.filter ?? {}) }
  delete f.concurrency
  return f
}

function formatKv(n) {
  if (n == null || !Number.isFinite(n)) return '—'
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  return `${Math.round(n)}`
}

function formatTick(metricKey, value) {
  if (metricKey === 'kv_bytes') return formatKv(value)
  return Number.isFinite(value) ? String(Math.round(value)) : '—'
}

function yAxisLabel(metricKey, metricLabel, unit) {
  if (metricKey === 'kv_bytes') return 'KV bytes'
  if (unit) return `${metricLabel} (${unit})`
  return metricLabel
}

function formatCellValue(metricKey, value) {
  if (value == null || !Number.isFinite(value)) return '—'
  if (metricKey === 'kv_bytes') {
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)} MB`
    if (value >= 1e3) return `${(value / 1e3).toFixed(0)} KB`
    return `${Math.round(value)} B`
  }
  return value.toFixed(1)
}

function TogglePill({ active, onClick, children }) {
  return (
    <button
      type="button"
      className={`toggle-pill${active ? ' active' : ''}`}
      aria-pressed={active}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

export default function DistributionBoxplot({ data, systemsById }) {
  const cfg = data.boxplot ?? {}

  const metrics = cfg.metrics ?? [
    { key: 'ttft_ms', label: 'TTFT', unit: 'ms' },
    { key: 'tbt_ms_p50', label: 'TBT', unit: 'ms' },
    { key: 'kv_bytes', label: 'KV bytes', unit: '' },
  ]

  const staticFilter = useMemo(() => baseFilterWithoutConcurrency(cfg), [cfg])

  const concurrencyOptions = useMemo(() => {
    const grid = data.source?.workload?.concurrency_grid
    if (Array.isArray(grid) && grid.length) {
      return [...new Set(grid)].sort((a, b) => a - b)
    }
    const fromRuns = (data.runs ?? [])
      .filter((r) => runMatchesFilter(r, staticFilter))
      .map((r) => r.concurrency ?? 1)
    return [...new Set(fromRuns)].sort((a, b) => a - b)
  }, [data.runs, data.source, staticFilter])

  const initialConcurrency =
    cfg.filter?.concurrency ??
    cfg.default_concurrency ??
    concurrencyOptions.find((c) => c > 1) ??
    concurrencyOptions[0] ??
    1

  const [metricKey, setMetricKey] = useState(cfg.metric ?? metrics[0]?.key ?? 'ttft_ms')
  const [concurrency, setConcurrency] = useState(initialConcurrency)

  const activeFilter = useMemo(
    () => ({ ...staticFilter, concurrency }),
    [staticFilter, concurrency],
  )

  const metricLabel =
    metrics.find((m) => m.key === metricKey)?.label ??
    (metricKey === 'ttft_ms' ? 'TTFT' : metricKey)
  const unit = metrics.find((m) => m.key === metricKey)?.unit ?? 'ms'
  const samplesField = samplesFieldForMetric(metricKey, cfg)
  const chartCaption = boxplotTitle(metricLabel, activeFilter)
  const yLabel = yAxisLabel(metricKey, metricLabel, unit)

  const groups = useMemo(() => {
    const filtered = (data.runs ?? []).filter((r) => runMatchesFilter(r, activeFilter))
    const bySystem = new Map()
    for (const run of filtered) {
      const id = run[cfg.group_field ?? 'system_id']
      const chunk = samplesForRun(run, samplesField, metricKey)
      if (!chunk.length) continue
      if (!bySystem.has(id)) bySystem.set(id, [])
      bySystem.get(id).push(...chunk)
    }
    const order = (data.systems ?? []).map((s) => s.id)
    const ids = [...bySystem.keys()].sort((a, b) => {
      const ia = order.indexOf(a)
      const ib = order.indexOf(b)
      if (ia === -1 && ib === -1) return a.localeCompare(b)
      if (ia === -1) return 1
      if (ib === -1) return -1
      return ia - ib
    })
    return ids.map((id) => ({
      id,
      label: systemsById[id]?.label ?? id,
      values: bySystem.get(id) ?? [],
      stats: boxFive(bySystem.get(id) ?? []),
    }))
  }, [
    data.runs,
    data.systems,
    activeFilter,
    cfg.group_field,
    samplesField,
    metricKey,
    systemsById,
  ])

  if (!data.boxplot) return null

  if (!groups.length || groups.every((g) => !g.stats)) {
    return (
      <p className="plot-error">
        No sample arrays for boxplot — add <code>{samplesField}</code> on runs matching{' '}
        <code>{JSON.stringify(activeFilter)}</code>.
      </p>
    )
  }

  const W = 720
  const height = 300
  const padL = 56
  const padR = 16
  const padT = 16
  const padB = 44
  const plotW = W - padL - padR
  const plotH = height - padT - padB
  const groupW = groups.length > 0 ? plotW / groups.length : plotW
  const boxW = Math.min(36, groupW * 0.45)

  const allValues = groups.flatMap((g) => g.values).filter(Number.isFinite)
  const yMax = allValues.length ? Math.max(...allValues) * 1.08 : 1
  const yMin = 0
  const yTicks = 5
  const tickVals = Array.from({ length: yTicks }, (_, i) => (yMax * i) / (yTicks - 1))

  const palette = ['#2563eb', '#ea580c', '#059669', '#9333ea', '#dc2626', '#0891b2']

  return (
    <figure className="metrics-plot boxplot-figure">
      <figcaption>{chartCaption}</figcaption>
      {cfg.subtitle ? <p className="section-lead">{cfg.subtitle}</p> : null}

      <div className="plot-controls" role="toolbar" aria-label="Boxplot settings">
        <div className="plot-control-group">
          <span className="plot-control-label">Distribution</span>
          {metrics.map((m) => (
            <TogglePill
              key={m.key}
              active={metricKey === m.key}
              onClick={() => setMetricKey(m.key)}
            >
              {m.label}
            </TogglePill>
          ))}
        </div>
        {concurrencyOptions.length > 0 ? (
          <div className="plot-control-group">
            <span className="plot-control-label">Concurrency</span>
            {concurrencyOptions.map((n) => (
              <TogglePill
                key={n}
                active={concurrency === n}
                onClick={() => setConcurrency(n)}
              >
                N={n}
              </TogglePill>
            ))}
          </div>
        ) : null}
      </div>

      <svg
        viewBox={`0 0 ${W} ${height}`}
        className="plot-svg"
        role="img"
        aria-label={chartCaption}
      >
        {tickVals.map((tv, ti) => {
          const y = scaleY(tv, yMin, yMax, plotH, padT)
          return (
            <g key={`${metricKey}-tick-${ti}`}>
              <line
                x1={padL}
                y1={y}
                x2={W - padR}
                y2={y}
                stroke="var(--border)"
                strokeDasharray="4 4"
                opacity={0.5}
              />
              <text x={padL - 6} y={y + 4} textAnchor="end" className="tick-label">
                {formatTick(metricKey, tv)}
              </text>
            </g>
          )
        })}
        <line
          x1={padL}
          y1={padT + plotH}
          x2={W - padR}
          y2={padT + plotH}
          stroke="var(--border)"
        />
        {groups.map((g, gi) => {
          const gx = padL + groupW * gi + groupW / 2
          const color = palette[gi % palette.length]
          return (
            <g key={g.id}>
              {g.stats ? (
                <BoxGlyph
                  stats={g.stats}
                  cx={gx}
                  boxW={boxW}
                  color={color}
                  yMin={yMin}
                  yMax={yMax}
                  plotH={plotH}
                  padT={padT}
                />
              ) : null}
              <text
                x={gx}
                y={padT + plotH + 16}
                textAnchor="middle"
                className="tick-label"
              >
                {g.label.length > 18 ? `${g.label.slice(0, 16)}…` : g.label}
              </text>
            </g>
          )
        })}
        <text
          x={14}
          y={padT + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${padT + plotH / 2})`}
          className="axis-label"
        >
          {yLabel}
        </text>
      </svg>

      <p className="boxplot-caption">
        Box: p25–p75; thick line: median; whiskers: furthest sample within 1.5×IQR; circles: outliers beyond that. Each box pools{' '}
        <strong>per-client</strong> samples across all repetitions in this cell (
        {groups.map((g) => `${g.label} n=${g.stats?.n ?? 0}`).join('; ')}).
      </p>

      <table className="plot-table boxplot-stats-table">
        <thead>
          <tr>
            <th>System</th>
            <th>n</th>
            <th>min</th>
            <th>p50</th>
            <th>p95</th>
            <th>max</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => {
            const s = g.values.slice().sort((a, b) => a - b)
            const p50 = percentile(s, 0.5)
            const p95 = percentile(s, 0.95)
            return (
              <tr key={g.id}>
                <td>{g.label}</td>
                <td>{s.length}</td>
                <td>{formatCellValue(metricKey, s.length ? s[0] : null)}</td>
                <td>{formatCellValue(metricKey, p50)}</td>
                <td>{formatCellValue(metricKey, p95)}</td>
                <td>{formatCellValue(metricKey, s.length ? s[s.length - 1] : null)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </figure>
  )
}
