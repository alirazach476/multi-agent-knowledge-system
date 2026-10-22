# Incident Runbook — RAG / Knowledge Services

## Severity
- **SEV-1**: Production Q&A unavailable or returning systematically ungrounded answers.
- **SEV-2**: Elevated latency (>5s p95) or partial index staleness.
- **SEV-3**: Single-tenant or non-production degradation.

## Escalation
1. Acknowledge in `#ai-platform-oncall` within 15 minutes.
2. Page the primary on-call via PagerDuty (`rag-platform` schedule).
3. If unresolved after 30 minutes, escalate to the platform tech lead.
4. Legal/privacy incidents escalate to Security IR immediately.

## First checks
- Confirm OpenAI / embedding provider status.
- Verify FAISS (or remote vector) index age vs last knowledge publish.
- Spot-check faithfulness on three golden questions before declaring recovery.
