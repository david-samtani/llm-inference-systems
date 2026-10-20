import { useCallback, useMemo, useState } from 'react'

const METRIC_OPTIONS = [
  { key: 'ttft_ms', label: 'TTFT' },
  { key: 'tbt_ms_p50', label: 'TBT p50' },
  { key: 'tbt_ms_mean', label: 'TBT mean' },
  { key: 'kv_bytes', label: 'KV bytes' },
]

const METRIC_KEYS = Object.fromEntries(
  METRIC_OPTIONS.map(({ key }) => [
    key,
    (run) => run[key],
  ]),
)

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

function median(values) {
  const nums = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b)
  if (!nums.length) return null
  const mid = Math.floor(nums.length / 2)
  if (nums.length % 2 === 0) return (nums[mid - 1] + nums[mid]) / 2
  return nums[mid]
}

function mean(values) {
  const nums = values.filter((v) => Number.isFinite(v))
  if (!nums.length) return null
  return nums.reduce((a, b) => a + b, 0) / nums.length
}

function runMatchesFilter(run, filter) {
  if (!filter) return true
  return Object.entries(filter).every(([key, val]) => run[key] === val)
}

function aggregateMetric(runs, metricKey, mode) {
  const pick = METRIC_KEYS[metricKey]
  if (!pick) return { y: null, raw: [] }
  const raw = runs.map(pick).filter((v) => Number.isFinite(v))
  if (!raw.length) return { y: null, raw: [] }
  const y = mode === 'mean' ? mean(raw) : median(raw)
  return { y, raw }
}

function formatKv(n) {
  if (n == null || !Number.isFinite(n)) return '—'
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`
  return `${Math.round(n)} B`
}

function formatMetric(metricKey, value) {
  if (value == null || !Number.isFinite(value)) return '—'
  if (metricKey === 'kv_bytes') return formatKv(value)
  return value.toFixed(1)
}

function metricSuffix(metricKey) {
  if (metricKey === 'kv_bytes') return ''
  return ' ms'
}

function yAxisLabelForMetric(metricKey, metricLabel) {
  if (metricKey === 'kv_bytes') return 'KV bytes'
  return `${metricLabel} (ms)`
}

function chartTitle(plot, metricLabel, aggregate) {
  const aggWord = aggregate === 'mean' ? 'Mean' : 'Median'
  const xName = plot.x_label ?? plot.x_field ?? 'x'
  const parts = Object.entries(plot.filter ?? {}).map(([k, v]) => `${k} = ${v}`)
  const filterSuffix = parts.length ? ` (${parts.join(', ')})` : ''
  return `${aggWord} ${metricLabel} vs ${xName}${filterSuffix}`
}

const PALETTE = [
  '#2563eb',
  '#ea580c',
  '#059669',
  '#9333ea',
  '#dc2626',
  '#0891b2',
]

export default function MetricsPlot({ data, systemsById }) {
  const { plot, runs } = data
  const defaultMetric = plot.metric ?? 'ttft_ms'

  const [metricKey, setMetricKey] = useState(defaultMetric)
  const [aggregate, setAggregate] = useState(plot.aggregate ?? 'median')
  const [yLog, setYLog] = useState(false)
  const [hidden, setHidden] = useState(() => new Set())
  const [hoverX, setHoverX] = useState(null)
  const [pinnedX, setPinnedX] = useState(null)

  const activeX = pinnedX ?? hoverX
  const pick = METRIC_KEYS[metricKey]

  const filtered = useMemo(
    () => runs.filter((r) => runMatchesFilter(r, plot.filter)),
    [runs, plot.filter],
  )

  const xValues = useMemo(
    () =>
      [...new Set(filtered.map((r) => r[plot.x_field]))].sort((a, b) => a - b),
    [filtered, plot.x_field],
  )

  const series = useMemo(() => {
    const seriesIds = [...new Set(filtered.map((r) => r[plot.series_field]))]
    return seriesIds.map((id) => {
      const points = xValues.map((x) => {
        const cellRuns = filtered.filter(
          (r) => r[plot.x_field] === x && r[plot.series_field] === id,
        )
        const { y, raw } = aggregateMetric(cellRuns, metricKey, aggregate)
        return { x, y, n: raw.length, raw }
      })
      return {
        id,
        label: systemsById[id]?.label ?? id,
        points,
      }
    })
  }, [
    filtered,
    xValues,
    plot.x_field,
    plot.series_field,
    metricKey,
    aggregate,
    systemsById,
  ])

  const visibleSeries = series.filter((s) => !hidden.has(s.id))

  const width = 720
  const height = 340
  const pad = { top: 28, right: 24, bottom: 48, left: 64 }
  const innerW = width - pad.left - pad.right
  const innerH = height - pad.top - pad.bottom

  const yVals = visibleSeries
    .flatMap((s) => s.points.map((p) => p.y))
    .filter((v) => Number.isFinite(v) && (!yLog || v > 0))

  const yMinLinear = 0
  const yMaxLinear = yVals.length ? Math.max(...yVals) * 1.08 : 1

  const yMinLog = yVals.length ? Math.min(...yVals) * 0.85 : 1
  const yMaxLog = yVals.length ? Math.max(...yVals) * 1.12 : 10

  const xMin = xValues[0] ?? 0
  const xMax = xValues[xValues.length - 1] ?? 1
  const xSpan = xMax - xMin || 1

  const xScale = useCallback(
    (x) => pad.left + ((x - xMin) / xSpan) * innerW,
    [xMin, xSpan, innerW],
  )

  const yScale = useCallback(
    (y) => {
      if (!Number.isFinite(y)) return pad.top + innerH
      if (yLog) {
        const lo = Math.log10(yMinLog)
        const hi = Math.log10(yMaxLog)
        const t = (Math.log10(y) - lo) / (hi - lo || 1)
        return pad.top + innerH - t * innerH
      }
      return (
        pad.top + innerH - ((y - yMinLinear) / (yMaxLinear - yMinLinear || 1)) * innerH
      )
    },
    [yLog, yMinLinear, yMaxLinear, yMinLog, yMaxLog, innerH],
  )

  const nearestX = useCallback(
    (svgX) => {
      if (!xValues.length) return null
      let best = xValues[0]
      let bestDist = Infinity
      for (const x of xValues) {
        const d = Math.abs(xScale(x) - svgX)
        if (d < bestDist) {
          bestDist = d
          best = x
        }
      }
      return best
    },
    [xValues, xScale],
  )

  const onPlotPointer = useCallback(
    (evt) => {
      const svg = evt.currentTarget
      const rect = svg.getBoundingClientRect()
      const scaleX = width / rect.width
      const svgX = (evt.clientX - rect.left) * scaleX
      if (svgX < pad.left || svgX > pad.left + innerW) {
        if (!pinnedX) setHoverX(null)
        return
      }
      setHoverX(nearestX(svgX))
    },
    [nearestX, pinnedX, innerW],
  )

  const onPlotClick = useCallback(
    (evt) => {
      const svg = evt.currentTarget
      const rect = svg.getBoundingClientRect()
      const scaleX = width / rect.width
      const svgX = (evt.clientX - rect.left) * scaleX
      if (svgX < pad.left || svgX > pad.left + innerW) return
      const x = nearestX(svgX)
      setPinnedX((prev) => (prev === x ? null : x))
      setHoverX(x)
    },
    [nearestX, innerW],
  )

  const toggleSeries = (id) => {
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const showAllSeries = () => setHidden(new Set())

  const yTicks = yLog ? 5 : 5
  const tickVals = useMemo(() => {
    if (yLog) {
      const lo = Math.log10(yMinLog)
      const hi = Math.log10(yMaxLog)
      return Array.from({ length: yTicks }, (_, i) => 10 ** (lo + ((hi - lo) * i) / (yTicks - 1)))
    }
    return Array.from(
      { length: yTicks },
      (_, i) => (yMaxLinear * i) / (yTicks - 1),
    )
  }, [yLog, yMinLog, yMaxLog, yMaxLinear, yTicks])

  const metricLabel =
    METRIC_OPTIONS.find((m) => m.key === metricKey)?.label ?? metricKey
  const yAxisLabel = yAxisLabelForMetric(metricKey, metricLabel)
  const chartCaption = chartTitle(plot, metricLabel, aggregate)

  if (!pick) {
    return <p className="plot-error">Unknown plot metric: {metricKey}</p>
  }

  return (
    <figure className="metrics-plot">
      <figcaption>{chartCaption}</figcaption>

      <div className="plot-controls" role="toolbar" aria-label="Chart settings">
        <div className="plot-control-group">
          <span className="plot-control-label">Metric</span>
          {METRIC_OPTIONS.map((m) => (
            <TogglePill
              key={m.key}
              active={metricKey === m.key}
              onClick={() => setMetricKey(m.key)}
            >
              {m.label}
            </TogglePill>
          ))}
        </div>
        <div className="plot-control-group">
          <span className="plot-control-label">Aggregate</span>
          <TogglePill
            active={aggregate === 'median'}
            onClick={() => setAggregate('median')}
          >
            median
          </TogglePill>
          <TogglePill
            active={aggregate === 'mean'}
            onClick={() => setAggregate('mean')}
          >
            mean
          </TogglePill>
        </div>
        <div className="plot-control-group">
          <span className="plot-control-label">Y axis</span>
          <TogglePill active={!yLog} onClick={() => setYLog(false)}>
            linear
          </TogglePill>
          <TogglePill active={yLog} onClick={() => setYLog(true)}>
            log
          </TogglePill>
        </div>
        {hidden.size > 0 ? (
          <button type="button" className="plot-link-btn" onClick={showAllSeries}>
            Show all lines ({hidden.size} hidden)
          </button>
        ) : null}
      </div>

      <div className="plot-stage">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={chartCaption}
          className="plot-svg"
          onMouseMove={onPlotPointer}
          onMouseLeave={() => {
            if (!pinnedX) setHoverX(null)
          }}
          onClick={onPlotClick}
        >
          <line
            x1={pad.left}
            y1={pad.top + innerH}
            x2={pad.left + innerW}
            y2={pad.top + innerH}
            stroke="var(--border)"
          />
          <line
            x1={pad.left}
            y1={pad.top}
            x2={pad.left}
            y2={pad.top + innerH}
            stroke="var(--border)"
          />
          {tickVals.map((tv) => (
            <g key={tv}>
              <line
                x1={pad.left - 4}
                y1={yScale(tv)}
                x2={pad.left}
                y2={yScale(tv)}
                stroke="var(--border)"
              />
              <text
                x={pad.left - 8}
                y={yScale(tv) + 4}
                textAnchor="end"
                className="tick-label"
              >
                {formatMetric(metricKey, tv)}
              </text>
              <line
                x1={pad.left}
                y1={yScale(tv)}
                x2={pad.left + innerW}
                y2={yScale(tv)}
                stroke="var(--border)"
                strokeDasharray="4 4"
                opacity={0.35}
              />
            </g>
          ))}
          <text
            x={pad.left + innerW / 2}
            y={height - 8}
            textAnchor="middle"
            className="axis-label"
          >
            {plot.x_label ?? plot.x_field}
          </text>
          <text
            x={16}
            y={pad.top + innerH / 2}
            textAnchor="middle"
            transform={`rotate(-90 16 ${pad.top + innerH / 2})`}
            className="axis-label"
          >
            {yAxisLabel}
          </text>
          {xValues.map((x) => (
            <g key={x}>
              <line
                x1={xScale(x)}
                y1={pad.top + innerH}
                x2={xScale(x)}
                y2={pad.top + innerH + 4}
                stroke="var(--border)"
              />
              <text
                x={xScale(x)}
                y={pad.top + innerH + 20}
                textAnchor="middle"
                className="tick-label"
              >
                {x}
              </text>
            </g>
          ))}

          {activeX != null ? (
            <line
              x1={xScale(activeX)}
              y1={pad.top}
              x2={xScale(activeX)}
              y2={pad.top + innerH}
              stroke="var(--accent)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
              pointerEvents="none"
            />
          ) : null}

          {series.map((s, si) => {
            if (hidden.has(s.id)) return null
            const color = PALETTE[si % PALETTE.length]
            const valid = s.points.filter((p) => p.y != null && Number.isFinite(p.y))
            if (valid.length < 1) return null
            const d =
              valid.length >= 2
                ? valid
                    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p.x)} ${yScale(p.y)}`)
                    .join(' ')
                : null
            return (
              <g key={s.id}>
                {d ? (
                  <path d={d} fill="none" stroke={color} strokeWidth={2.5} pointerEvents="none" />
                ) : null}
                {valid.map((p) => {
                  const highlighted = activeX === p.x
                  return (
                    <circle
                      key={`${s.id}-${p.x}`}
                      cx={xScale(p.x)}
                      cy={yScale(p.y)}
                      r={highlighted ? 6 : 4}
                      fill={color}
                      stroke={highlighted ? 'var(--text-h)' : 'none'}
                      strokeWidth={highlighted ? 1.5 : 0}
                      pointerEvents="none"
                    />
                  )
                })}
              </g>
            )
          })}

          <rect
            x={pad.left}
            y={pad.top}
            width={innerW}
            height={innerH}
            fill="transparent"
            style={{ cursor: 'crosshair' }}
          />
        </svg>

        {activeX != null ? (
          <div
            className={`plot-tooltip${pinnedX != null ? ' pinned' : ''}`}
            role="status"
          >
            <div className="plot-tooltip-head">
              <strong>
                {plot.x_label ?? plot.x_field}: {activeX}
              </strong>
              <span className="plot-tooltip-metric">
                {metricLabel}
                {metricSuffix(metricKey)}
                {' · '}
                {aggregate}
              </span>
              {pinnedX != null ? (
                <span className="plot-tooltip-hint">Pinned — click chart to unpin</span>
              ) : (
                <span className="plot-tooltip-hint">Click to pin</span>
              )}
            </div>
            <ul>
              {series.map((s, si) => {
                if (hidden.has(s.id)) return null
                const pt = s.points.find((p) => p.x === activeX)
                if (!pt || pt.y == null) return null
                const color = PALETTE[si % PALETTE.length]
                const rawStr =
                  pt.raw.length > 1
                    ? ` (${pt.raw.map((v) => formatMetric(metricKey, v)).join(', ')})`
                    : ''
                return (
                  <li key={s.id}>
                    <span className="swatch" style={{ background: color }} />
                    <span className="plot-tooltip-series">{s.label}</span>
                    <span className="plot-tooltip-value">
                      {formatMetric(metricKey, pt.y)}
                      {metricSuffix(metricKey)}
                    </span>
                    <span className="plot-tooltip-n">
                      n={pt.n}
                      {rawStr}
                    </span>
                  </li>
                )
              })}
            </ul>
          </div>
        ) : (
          <p className="plot-hover-hint">Hover or click the chart to inspect values.</p>
        )}
      </div>

      <ul className="plot-legend">
        {series.map((s, si) => {
          const isHidden = hidden.has(s.id)
          return (
            <li key={s.id}>
              <button
                type="button"
                className={`legend-toggle${isHidden ? ' off' : ''}`}
                aria-pressed={!isHidden}
                onClick={() => toggleSeries(s.id)}
                title={isHidden ? 'Show line' : 'Hide line'}
              >
                <span
                  className="swatch"
                  style={{
                    background: PALETTE[si % PALETTE.length],
                    opacity: isHidden ? 0.35 : 1,
                  }}
                />
                {s.label}
              </button>
            </li>
          )
        })}
      </ul>

      <table className="plot-table">
        <thead>
          <tr>
            <th>{plot.x_label ?? plot.x_field}</th>
            {series.map((s, si) => (
              <th key={s.id} className={hidden.has(s.id) ? 'col-hidden' : ''}>
                {s.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {xValues.map((x) => (
            <tr key={x} className={activeX === x ? 'row-active' : ''}>
              <td>{x}</td>
              {series.map((s) => {
                const pt = s.points.find((p) => p.x === x)
                return (
                  <td key={s.id} className={hidden.has(s.id) ? 'col-hidden' : ''}>
                    {formatMetric(metricKey, pt?.y ?? null)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  )
}
