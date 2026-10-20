import './App.css'
import Hideable from './Hideable.jsx'
import DistributionBoxplot from './DistributionBoxplot.jsx'
import MetricsPlot from './MetricsPlot.jsx'
import DATA from './results.js'

function systemsById(systems) {
  return Object.fromEntries(systems.map((s) => [s.id, s]))
}

function App() {
  const meta = DATA.meta
  const systems = DATA.systems ?? []
  const byId = systemsById(systems)

  return (
    <div className="App">
      <header className="report-header">
        <p className="eyebrow">LLM serving benchmark</p>
        <h1>{meta.title}</h1>
        <p className="meta-line">
          {meta.name} · @{meta.github} · {meta.date}
        </p>
        <p className="meta-line subtle">
          Model: {meta.model} · {meta.device}
        </p>
      </header>

      <main className="report-main">
        <section className="callout takeaways" id="takeaways">
          <h2>Takeaways</h2>
          <ol>
            {(DATA.takeaways ?? []).map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ol>
        </section>

        <section className="callout summary" id="summary">
          <h2>Summary</h2>
          <p>{DATA.summary}</p>
        </section>

        <Hideable title="Results" sectionId="results" defaultOpen>
          <p className="section-lead">
            Use the pills above each chart to switch metric, aggregation and concurrency.
            Click a legend entry to hide a system, and hover or click a point to see or pin
            its values.
          </p>
          <h3 className="results-subhead">1. Line chart</h3>
          <MetricsPlot data={DATA} systemsById={byId} />
          <h3 className="results-subhead">2. Boxplot</h3>
          {DATA.boxplot ? (
            <DistributionBoxplot data={DATA} systemsById={byId} />
          ) : (
            <p className="plot-error">
              Missing top-level <code>boxplot</code> in your results JSON (required).
              Add <code>ttft_ms_samples</code> / <code>tbt_ms_samples</code> on concurrent
              runs — see README.
            </p>
          )}
        </Hideable>

        <Hideable title="Systems compared" sectionId="systems" defaultOpen={false}>
          <ul className="systems-list">
            {systems.map((s) => (
              <li key={s.id}>
                <strong>{s.label}</strong>{' '}
                <span className="pill">{s.family}</span>
                <p>{s.description}</p>
                {s.repo ? (
                  <p className="system-repo">
                    <a href={s.repo} target="_blank" rel="noreferrer">
                      {s.repo.replace(/^https:\/\//, '')}
                    </a>
                  </p>
                ) : null}
                {s.novelty ? (
                  <p className="system-novelty">
                    <strong>Key novelty:</strong> {s.novelty}
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        </Hideable>

        <Hideable title="Measurement setup" sectionId="setup" defaultOpen={false}>
          <dl className="setup-dl">
            <dt>Harness</dt>
            <dd>
              <code>{DATA.source.harness}</code>
            </dd>
            <dt>Results JSON</dt>
            <dd>
              <code>{DATA.source.json_file}</code>
            </dd>
            <dt>Wall clock</dt>
            <dd>
              {DATA.source.started} → {DATA.source.finished}
            </dd>
            <dt>Workload</dt>
            <dd>
              Prompt targets{' '}
              {DATA.source.workload.prompt_token_targets.join(', ')} tokens ·{' '}
              {DATA.source.workload.max_new_tokens} new tokens · concurrency{' '}
              {DATA.source.workload.concurrency_grid.join(', ')} ·{' '}
              {DATA.source.workload.reps_per_cell} reps/cell
            </dd>
          </dl>
          <p className="section-lead">{DATA.source.workload.notes}</p>
        </Hideable>

        <Hideable title="Discussion" sectionId="discussion" defaultOpen={false}>
          <h3 className="results-subhead">Baseline: minimal_full_server (no scheduler)</h3>
          <p>
            At N = 1 the baseline is competitive on TTFT (18 / 23 / 92 ms at 128 / 512 /
            1984 tokens) because prefill on a 1.1B model is one fast forward pass. Its
            weaknesses are decode and concurrency. TBT is ~15.8 ms because every step is a
            separate, un-batched Hugging Face forward pass with Python overhead and no CUDA
            graphs. At N = 4 the TTFT boxplot is a staircase: clients wait roughly 0, 1, 2
            and 3 s, because each 64-token generation (~1 s) must finish before the next
            starts. The median is ~1.5 s, but p95 and max reach 3.0-3.4 s, set entirely by
            queue position. Active requests maintain consistent TBT, but sequential
            execution causes head-of-line blocking for waiting clients. A throughput
            number alone would hide this 3 s tail.
          </p>

          <h3 className="results-subhead">vLLM: PagedAttention + continuous batching</h3>
          <p>
            Continuous batching removes the staircase. All four requests are scheduled
            together, so median TTFT at N = 4 is 26 / 72 /
            191 ms instead of ~1.5 s. At 1984 tokens the per-client spread is still visible
            (min 77 ms, p95 299 ms, max 301 ms), most likely because four long prefills do
            not fit in one scheduling step. Decode batching is nearly free. TBT only rises
            from 5.6 to 6.0 ms at N = 4, because decode is memory-bandwidth bound and one
            read of the weights serves all four sequences. My KV-bytes column uses the same
            dense formula for every system, so PagedAttention shows up in the block math
            instead. With 16-token blocks, a 128-token request (191 tokens after decoding)
            occupies 12 blocks, 4.33 MB versus 4.30 MB logical. Reserving the full
            2048-token context per request would cost 46.1 MB, 10.7x more. That headroom
            is what lets vLLM admit many concurrent requests. This workload never fills the
            A10, so it appears as spare memory rather than lower latency.
          </p>

          <h3 className="results-subhead">Mooncake store: cold vs warm prefix</h3>
          <p>
            Mooncake only changes how prefill KV is obtained, so TBT and KV bytes match
            vLLM and its whole effect is in TTFT. Cold, it costs about 5 ms more than vLLM
            at N = 1 and roughly double at N = 4 and 1984 tokens (377 vs 191 ms, p95 460
            vs 299 ms), most likely because every prefill's KV is also written to the store
            over TCP during the burst. Warm, the prefix is fetched instead of recomputed. At
            1984 tokens TTFT falls to 43 ms at N = 1 (77 ms cold, 73 ms vLLM) and 142 ms at
            N = 4, with p95 182 ms, the lowest tail of any system. The gain grows with
            prompt length (1, 3 and 35 ms at 128, 512 and 1984 tokens) because the saved
            work is prefill while the fetch cost is roughly fixed. The vLLM log confirms the
            hits came from Mooncake, since vLLM's own prefix cache was off. The external
            hit rate reached 46% in the WildChat warm run, close to the ~50% expected when
            each prompt is sent once to fill the store and once to be timed.
          </p>

          <h3 className="results-subhead">Real prompts (WildChat-1M sanity check)</h3>
          <p>
            On 20 real first-turn prompts (mostly under 100 tokens, longest 439) the
            scheduling result holds. At N = 4 the baseline's median TTFT is 1.5 s with a
            3.0 s max, against a 21 ms median and 27 ms max for vLLM. The prefix-reuse
            result flips. Warm Mooncake is no faster than cold (20 vs 21 ms at N = 1, 31 vs
            23 ms median at N = 4), so reuse pays off for long shared context such as
            system prompts, documents and chat history, not one-off chats.
          </p>

          <h3 className="results-subhead">Limitations</h3>
          <p>
            One 1.1B model on one GPU makes prefill cheap, so absolute gaps, and
            Mooncake's benefit, are small. A larger model could increase the value of
            prefix reuse, depending on compute and transfer costs. The Mooncake
            store ran on the same machine over TCP, not across a cluster with RDMA. Warm
            runs are the best case (the entire prompt cached), and concurrency stops at 4.
            Synthetic prompts of the same length share a ~16-token stem (see Measurement
            setup). Timing is not perfectly symmetric. The baseline is called in-process,
            so it skips the HTTP and streaming overhead included for the vLLM-based
            systems, which if anything understates vLLM's advantage.
          </p>

          <p style={{ marginTop: '2rem' }}>
            Source:{' '}
            <a href={meta.repo_url} target="_blank" rel="noreferrer">
              {meta.repo_url.replace(/^https:\/\//, '')}
            </a>
          </p>
        </Hideable>

        <Hideable title="Raw run count" sectionId="raw" defaultOpen={false}>
          <p>
            {(DATA.runs ?? []).length} rows in <code>runs[]</code> (commit the
            full JSON; the page aggregates medians for the plot).
          </p>
        </Hideable>
      </main>
    </div>
  )
}

export default App
