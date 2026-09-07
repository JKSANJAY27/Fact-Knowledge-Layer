"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import "./globals.css";

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS — 4 required case example prompts
// ─────────────────────────────────────────────────────────────────────────────
const EXAMPLE_PROMPTS = [
  {
    id: "corroborated",
    caseLabel: "Corroborated Fact",
    icon: "✓",
    tagClass: "tag-corroborated",
    question:
      "What is Delhivery's Adjusted EBITDA and does it appear consistently across the annual report and earnings presentation?",
    hint: "Tests cross-document corroboration — same metric, same entity, two different documents",
  },
  {
    id: "contradicted",
    caseLabel: "Contradiction",
    icon: "⚡",
    tagClass: "tag-contradicted",
    question:
      "What revenue figures does the earnings presentation report for Delhivery across FY2022 and FY2024 — are they consistent?",
    hint: "Tests contradiction detection — Revenue 46 Cr (FY2022) vs 5 Cr (FY2024) — period-split contradiction",
  },
  {
    id: "contextual",
    caseLabel: "Context-Explained Discrepancy",
    icon: "⧗",
    tagClass: "tag-contextual",
    question:
      "Delhivery's PIN code reach appears as different numbers across documents — what is the correct figure and how can the discrepancy be explained?",
    hint: "Tests context reconciliation — 4,445 pin codes (total) vs 700 pin codes (new tier-2/3 expansion)",
  },
  {
    id: "abstention",
    caseLabel: "Abstention / Failure",
    icon: "?",
    tagClass: "tag-abstention",
    question:
      "What is the exact unit and measurement basis for India's GDP growth rate or inflation figures from the Economic Survey?",
    hint: "Tests abstention-over-hallucination — India macroeconomic figures parsed but units/context unclear",
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────
function confidenceClass(conf) {
  if (conf >= 0.75) return "conf-high";
  if (conf >= 0.45) return "conf-med";
  return "conf-low";
}

function confidenceLabel(conf) {
  if (conf >= 0.75) return `${Math.round(conf * 100)}% confidence`;
  if (conf >= 0.45) return `${Math.round(conf * 100)}% — moderate`;
  return `${Math.round(conf * 100)}% — low`;
}

function relationBadge(type) {
  const MAP = {
    corroborated: { label: "Corroborated", cls: "tag-corroborated", icon: "✓" },
    contradicted: { label: "Contradiction", cls: "tag-contradicted", icon: "⚡" },
    contextual: { label: "Context-Resolved", cls: "tag-contextual", icon: "⧗" },
    abstention: { label: "Abstention", cls: "tag-abstention", icon: "?" },
    unresolved: { label: "Unresolved", cls: "tag-abstention", icon: "~" },
  };
  return MAP[type?.toLowerCase()] || { label: type || "—", cls: "tag-abstention", icon: "·" };
}

function shortFilename(name = "") {
  return name.replace(/\d{2}-/, "").replace(/\.pdf$/i, "").replace(/-/g, " ");
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export default function Home() {
  // ── Stats & documents
  const [stats, setStats] = useState({ total_documents: 0, total_facts: 0, total_pages: 0, relationships: {}, failures: {} });
  const [documents, setDocuments] = useState([]);

  // ── Chat
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // ── PDF viewer
  const [pdfPanel, setPdfPanel] = useState(null); // { filename, page, docName }

  // ── Upload
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const uploadJobRef = useRef(null);
  const pollRef = useRef(null);

  // ── Session ID for temporary evaluator uploads
  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      let sid = sessionStorage.getItem("fkl_session_id");
      if (!sid) {
        sid = "sess_" + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
        sessionStorage.setItem("fkl_session_id", sid);
      }
      setSessionId(sid);
    }
  }, []);

  // ── Auto-cleanup uploaded session docs on tab/window close
  useEffect(() => {
    if (!sessionId) return;
    const cleanup = () => {
      try {
        navigator.sendBeacon(`/api/session/${sessionId}`, new Blob([], { type: "text/plain" }));
      } catch (_) {}
    };
    window.addEventListener("beforeunload", cleanup);
    return () => window.removeEventListener("beforeunload", cleanup);
  }, [sessionId]);

  // ── Fetch stats + docs on mount
  useEffect(() => {
    refreshStats();
  }, []);

  const refreshStats = useCallback(async () => {
    try {
      const [s, d] = await Promise.all([
        fetch("/api/stats").then((r) => r.json()).catch(() => ({})),
        fetch("/api/documents").then((r) => r.json()).catch(() => []),
      ]);
      setStats(s || {});
      setDocuments(d || []);
    } catch (_) {}
  }, []);

  // ── Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ── Textarea auto-resize
  const handleInputChange = (e) => {
    setInput(e.target.value);
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
    }
  };

  // ── Send message
  const sendMessage = useCallback(async (questionOverride) => {
    const question = (questionOverride || input).trim();
    if (!question || loading) return;

    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const userMsg = { role: "user", text: question };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // API field is 'query' not 'question'
        body: JSON.stringify({ query: question, top_k: 8 }),
      });
      const raw = await res.json();
      // Normalize API response to internal shape
      const data = {
        answer: raw.answer || "",
        citations: (raw.grounded_facts || []).map((f) => ({
          fact_id: f.fact_id,
          filename: f.document_name,
          document_id: f.fact_id,
          page_number: f.page_number,
          entity: f.entity,
          metric: f.metric,
          raw_value: f.raw_value,
          scope: f.scope,
          period: f.canonical_period,
          quote: f.quoted_text,
        })),
        relationships: (raw.related_reconciliations || []).map((r) => ({
          relation_type: r.relation_type,
          explanation: r.explanation,
          difference_field: r.difference_field,
        })),
        confidence: 0,
        abstained: raw.confidence_note?.toLowerCase().includes("abstain") || false,
        abstention_reason: raw.confidence_note?.toLowerCase().includes("abstain") ? raw.confidence_note : "",
        failures: [],
        confidence_note: raw.confidence_note || "",
      };
      const assistantMsg = { role: "assistant", data };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          data: {
            answer: "⚠️ Could not connect to the backend. Make sure the FastAPI server is running on port 8000.",
            citations: [],
            relationships: [],
            abstained: false,
            failures: [],
          },
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ── Open PDF citation viewer
  const openPdf = (filename, page, docName) => {
    // Build URL for the PDF served at /static/pdfs/
    setPdfPanel({ filename, page: page || 1, docName: docName || filename });
  };

  const closePdf = () => setPdfPanel(null);

  // ── File upload handler
  const handleUpload = async (file) => {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus("Only PDF files are supported.");
      return;
    }

    setUploading(true);
    setUploadProgress(5);
    setUploadStatus(`Uploading ${file.name}…`);

    const fd = new FormData();
    fd.append("file", file);
    if (sessionId) {
      fd.append("session_id", sessionId);
    }

    try {
      const res = await fetch("/api/documents/upload", { method: "POST", body: fd });
      const { job_id, filename } = await res.json();

      uploadJobRef.current = job_id;
      setUploadStatus(`Processing ${filename}…`);

      // Poll progress
      pollRef.current = setInterval(async () => {
        try {
          const pRes = await fetch(`/api/documents/upload/status/${job_id}`);
          const pData = await pRes.json();
          setUploadProgress(pData.progress || 0);
          setUploadStatus(
            pData.status === "done"
              ? `✓ Done! ${pData.result?.facts_extracted || 0} facts extracted.`
              : `${pData.status}… ${pData.progress || 0}%`
          );

          if (pData.status === "done") {
            clearInterval(pollRef.current);
            setUploading(false);
            refreshStats();
          }
        } catch (_) {}
      }, 1500);
    } catch (err) {
      setUploadStatus("Upload failed. Please try again.");
      setUploading(false);
    }
  };

  const handleDropOrPick = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer ? e.dataTransfer.files[0] : e.target.files[0];
    if (file) handleUpload(file);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER HELPERS
  // ─────────────────────────────────────────────────────────────────────────

  // Render a single assistant message
  const renderAssistantMessage = (data, idx) => {
    if (!data) return null;

    const {
      answer = "",
      citations = [],
      confidence = 0,
      relationships = [],
      failures = [],
      abstained = false,
      abstention_reason = "",
    } = data;

    // Helper to render text with inline clickable citation pills
    const renderAnswerText = (text = "") => {
      if (!text) return null;

      // Matches citations inside parentheses: (doc_name, p. 5) or (Annual Report FY24, p. 2)
      const citationRegex = /\(([^()\n]*?(?:p\.|page|Page)\s*(\d+)[^()\n]*?)\)/g;
      const elements = [];
      let lastIndex = 0;
      let match;

      while ((match = citationRegex.exec(text)) !== null) {
        const matchStart = match.index;
        const matchEnd = match.index + match[0].length;
        const innerContent = match[1];
        const pageNum = parseInt(match[2], 10);

        if (matchStart > lastIndex) {
          elements.push(text.substring(lastIndex, matchStart));
        }

        const innerLower = innerContent.toLowerCase();

        // Match citation from citations array
        let targetDoc = citations.find((c) => {
          if (c.page_number !== pageNum) return false;
          const fn = (c.filename || "").toLowerCase();
          return innerLower.split(/[\s,._-]+/).some((tok) => tok.length > 2 && fn.includes(tok));
        });

        if (!targetDoc) {
          targetDoc = citations.find((c) => c.page_number === pageNum);
        }

        if (!targetDoc) {
          targetDoc = citations.find((c) => {
            const fn = (c.filename || "").toLowerCase();
            return innerLower.split(/[\s,._-]+/).some((tok) => tok.length > 3 && fn.includes(tok));
          });
        }

        if (!targetDoc) {
          const found = documents.find((d) => {
            const fn = (d.filename || "").toLowerCase();
            return innerLower.split(/[\s,._-]+/).some((tok) => tok.length > 3 && fn.includes(tok));
          });
          if (found) targetDoc = { filename: found.filename, page_number: pageNum };
        }

        const filename = targetDoc?.filename || (innerContent.replace(/,\s*p\..*/i, "").trim().replace(/\s+/g, "-") + ".pdf");
        const page = targetDoc?.page_number || pageNum || 1;
        const shortName = shortFilename(filename);

        elements.push(
          <button
            key={`cite-${matchStart}`}
            className="inline-citation-btn"
            onClick={() => openPdf(filename, page, shortName)}
            title={`Open ${filename} at page ${page}`}
          >
            <span className="inline-chip-icon">📄</span>
            <span>{shortName}</span>
            <span className="inline-chip-page">p.{page}</span>
          </button>
        );

        lastIndex = matchEnd;
      }

      if (lastIndex < text.length) {
        elements.push(text.substring(lastIndex));
      }

      return elements;
    };

    const hasInlineCitations = /\(([^()\n]*?(?:p\.|page|Page)\s*(\d+)[^()\n]*?)\)/i.test(answer);

    // Detect answer type from displayed relationships to ensure header matches content
    const displayedRels = relationships?.slice(0, 3) || [];
    const hasAbstention = abstained || failures?.length > 0;
    const hasContradiction = displayedRels.some((r) =>
      r.relation_type?.toLowerCase() === "contradicted"
    );
    const hasContextual = displayedRels.some((r) =>
      r.relation_type?.toLowerCase()?.includes("contextual")
    );
    const hasCorroboration = displayedRels.some((r) =>
      r.relation_type?.toLowerCase() === "corroborated"
    );

    const sectionClass = hasAbstention
      ? "section-abstention"
      : hasContradiction
      ? "section-contradicted"
      : hasContextual
      ? "section-contextual"
      : hasCorroboration
      ? "section-corroborated"
      : "";

    const headerClass = hasAbstention
      ? "header-abstention"
      : hasContradiction
      ? "header-contradicted"
      : hasContextual
      ? "header-contextual"
      : hasCorroboration
      ? "header-corroborated"
      : "";

    const sectionLabel = hasAbstention
      ? "Abstention Logged"
      : hasContradiction
      ? "Contradiction Detected"
      : hasContextual
      ? "Context-Resolved Discrepancy"
      : hasCorroboration
      ? "Cross-Document Corroboration"
      : null;

    const sectionIcon = hasAbstention ? "?" : hasContradiction ? "⚡" : hasContextual ? "⧗" : hasCorroboration ? "✓" : null;

    return (
      <div key={idx} className="message-row">
        <div className="msg-avatar avatar-assistant">FK</div>
        <div className="bubble-wrap">
          <div className="chat-bubble bubble-assistant">
            {/* Main answer text with inline clickable citations */}
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.8 }}>
              {renderAnswerText(answer)}
            </div>

            {/* Relationship type banner */}
            {sectionLabel && (
              <div className={`answer-section ${sectionClass}`} style={{ marginTop: "0.75rem" }}>
                <div className={`answer-section-header ${headerClass}`}>
                  <span>{sectionIcon}</span>
                  <span>{sectionLabel}</span>
                  {confidence > 0 && (
                    <span className={`confidence-badge ${confidenceClass(confidence)}`}>
                      {confidenceLabel(confidence)}
                    </span>
                  )}
                </div>

                {/* Relationship details */}
                {displayedRels.length > 0 && (
                  <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: "0.3rem" }}>
                    {displayedRels.map((r, ri) => {
                      const b = relationBadge(r.relation_type);
                      return (
                        <div key={ri} style={{ marginBottom: "0.25rem", display: "flex", gap: "0.4rem", alignItems: "flex-start" }}>
                          <span className={`example-tag ${b.cls}`} style={{ flexShrink: 0 }}>
                            {b.icon} {b.label}
                          </span>
                          <span>{r.explanation || r.evidence_summary || ""}</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Abstention reason */}
                {hasAbstention && abstention_reason && (
                  <div style={{ fontSize: "0.78rem", color: "var(--color-abstain)", marginTop: "0.3rem" }}>
                    ⚠ {abstention_reason}
                  </div>
                )}

                {failures?.length > 0 && !abstention_reason && (
                  <div style={{ fontSize: "0.78rem", color: "var(--color-abstain)", marginTop: "0.3rem" }}>
                    {failures.slice(0, 2).map((f, fi) => (
                      <div key={fi}>⚠ {f.reason || f.failure_type}: {f.raw_text?.slice(0, 80)}…</div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Bottom Citations (shown only if not already cited inline) */}
            {!hasInlineCitations && citations?.length > 0 && (
              <div className="citations-row">
                {citations.slice(0, 8).map((c, ci) => {
                  const short = shortFilename(c.filename || c.document_id || "");
                  return (
                    <button
                      key={ci}
                      className="citation-chip"
                      onClick={() => openPdf(c.filename || `${c.document_id}.pdf`, c.page_number, short)}
                      title={`Open ${c.filename} at page ${c.page_number}`}
                    >
                      <span className="chip-icon">📄</span>
                      <span>{short || c.document_id}</span>
                      <span className="chip-page">p.{c.page_number}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // ─────────────────────────────────────────────────────────────────────────
  // STATS BAR — navbar numbers
  // ─────────────────────────────────────────────────────────────────────────
  const corroboratedCount = Object.values(stats.relationships || {}).reduce((a, b) => a + b, 0);
  const failCount = Object.values(stats.failures || {}).reduce((a, b) => a + b, 0);

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      {/* ── NAVBAR ── */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">FK</div>
          <div>
            <span className="brand-title">Fact Knowledge Layer</span>
            <span className="brand-sub">Financial Intelligence</span>
          </div>
        </div>
        <div className="nav-stats">
          <div className="nav-dot" title="System live" />
          <div className="nav-stat"><strong>{stats.total_documents || 0}</strong> docs</div>
          <div className="nav-stat"><strong>{stats.total_pages || 0}</strong> pages</div>
          <div className="nav-stat"><strong>{stats.total_facts || 0}</strong> facts</div>
          <div className="nav-stat"><strong>{corroboratedCount}</strong> relationships</div>
          <div className="nav-stat"><strong>{failCount}</strong> abstentions</div>
        </div>
      </nav>

      {/* ── MAIN ── */}
      <div className="main-layout">
        {/* ── LEFT SIDEBAR ── */}
        <aside className="sidebar">
          {/* 4 Case Example Prompts */}
          <div className="sidebar-section">
            <div className="sidebar-label">Example Queries</div>
            <div className="example-prompts">
              {EXAMPLE_PROMPTS.map((p) => (
                <button
                  key={p.id}
                  className="example-btn"
                  data-case={p.id}
                  onClick={() => sendMessage(p.question)}
                  disabled={loading}
                  title={p.hint}
                >
                  <div className={`example-tag ${p.tagClass}`}>
                    {p.icon} {p.caseLabel}
                  </div>
                  <span className="example-question">{p.question}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Upload Panel */}
          <div className="sidebar-section">
            <div className="sidebar-label">Upload New PDF</div>
            <div
              className={`upload-zone${dragOver ? " drag-over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDropOrPick}
            >
              <input
                type="file"
                accept=".pdf"
                onChange={handleDropOrPick}
                disabled={uploading}
              />
              <div className="upload-icon">📂</div>
              <div className="upload-text">
                {uploading ? "Processing…" : "Drop PDF here or click to browse"}
              </div>
              <div className="upload-hint">Schema-free · Session-scoped (auto-removed on tab close)</div>
            </div>
            {(uploadStatus || uploading) && (
              <>
                <div className="upload-progress-bar">
                  <div className="upload-progress-fill" style={{ width: `${uploadProgress}%` }} />
                </div>
                <div className="upload-status">{uploadStatus}</div>
              </>
            )}
          </div>

          {/* Documents list */}
          <div className="sidebar-section" style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <div className="sidebar-label">
              Ingested Documents
              {documents.length > 0 && <span className="facts-count-badge">{documents.length}</span>}
            </div>
            <div className="docs-list">
              {documents.length === 0 ? (
                <div className="empty-state">No documents ingested yet.<br />Upload a PDF or run ingestion.</div>
              ) : (
                documents.map((doc) => (
                  <div
                    key={doc.document_id}
                    className="doc-item"
                    onClick={() => openPdf(doc.filename, 1, shortFilename(doc.filename))}
                    title={`Open ${doc.filename}`}
                  >
                    <span className="doc-icon">📄</span>
                    <div>
                      <div className="doc-name">{shortFilename(doc.filename)}</div>
                      <div className="doc-meta">
                        {doc.page_count || "?"} pages
                        <span className="dot-sep">·</span>
                        {doc.document_id?.slice(0, 10)}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>

        {/* ── CHAT AREA ── */}
        <section className="chat-area">
          <div className="chat-messages">
            {messages.length === 0 ? (
              /* Welcome screen */
              <div className="welcome-screen">
                <div className="welcome-icon">🧠</div>
                <h1 className="welcome-title">Fact Knowledge Layer</h1>
                <p className="welcome-sub">
                  Ask any question about ingested financial and institutional documents.
                  Every answer is grounded in atomic facts with source citations — click any
                  citation to open the original PDF at the referenced page.
                </p>
                <div className="welcome-prompts">
                  {EXAMPLE_PROMPTS.map((p) => (
                    <button
                      key={p.id}
                      className="welcome-prompt-btn"
                      onClick={() => sendMessage(p.question)}
                    >
                      <span className={`welcome-prompt-tag ${p.tagClass}`}>
                        {p.icon} {p.caseLabel}
                      </span>
                      {p.question}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg, idx) => {
                if (msg.role === "user") {
                  return (
                    <div key={idx} className="message-row user">
                      <div className="msg-avatar avatar-user">U</div>
                      <div className="bubble-wrap">
                        <div className="chat-bubble bubble-user">{msg.text}</div>
                      </div>
                    </div>
                  );
                }
                return renderAssistantMessage(msg.data, idx);
              })
            )}

            {/* Typing indicator */}
            {loading && (
              <div className="message-row">
                <div className="msg-avatar avatar-assistant">FK</div>
                <div className="bubble-wrap">
                  <div className="chat-bubble bubble-assistant">
                    <div className="typing-indicator">
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input bar */}
          <div className="chat-input-bar">
            <div className="input-row">
              <textarea
                ref={textareaRef}
                className="chat-textarea"
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about any ingested document…"
                rows={1}
                disabled={loading}
              />
              <button
                className="send-btn"
                onClick={() => sendMessage()}
                disabled={loading || !input.trim()}
                title="Send (Enter)"
              >
                ↑
              </button>
            </div>
            <div className="input-hint">
              Press Enter to send · Shift+Enter for new line · Citations are clickable → opens PDF at source page
            </div>
          </div>
        </section>

        {/* ── PDF VIEWER PANEL ── */}
        <aside className={`pdf-panel${pdfPanel ? "" : " hidden"}`}>
          {pdfPanel && (
            <>
              <div className="pdf-panel-header">
                <div>
                  <div className="pdf-panel-title">
                    <span>📄</span>
                    <span>{pdfPanel.docName}</span>
                  </div>
                  <div className="pdf-panel-subtitle">Page {pdfPanel.page}</div>
                </div>
                <button className="pdf-close-btn" onClick={closePdf} title="Close PDF viewer">✕</button>
              </div>
              <div className="pdf-frame-container">
                <iframe
                  key={`${pdfPanel.filename}-${pdfPanel.page}`}
                  src={`/static/pdfs/${encodeURIComponent(pdfPanel.filename)}#page=${pdfPanel.page}`}
                  title={`${pdfPanel.docName} - Page ${pdfPanel.page}`}
                />
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
