import { useState } from 'react'

export default function Hideable({
  title,
  sectionId,
  defaultOpen = true,
  children,
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <section className="report-section" id={sectionId}>
      <button
        type="button"
        className="section-toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className={`chevron${open ? ' open' : ''}`} aria-hidden>
          ›
        </span>
        <h2>{title}</h2>
      </button>
      {open ? <div className="section-body">{children}</div> : null}
    </section>
  )
}
