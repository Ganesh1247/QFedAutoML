# QFedAutoML — Documentation Generation Prompt

Paste this into your AI IDE (same repo context as before — it needs to read your actual `backend/` and `frontend/` code to fill this in accurately, not invent it).

---

```
ROLE: You are documenting an already-built, already-tested project called
QFedAutoML (56/56 tests passing, running locally at frontend:3000,
backend:8000). Your job is NOT to redesign anything — it is to read the
actual source code in this repo and produce accurate, complete
documentation of every section of the system as it truly exists right now.

HARD RULE: Every claim in the documentation must be traceable to a real
file, route, function signature, or component in this repo. If something
described in the original architecture doc (docs/architecture.md) was
NOT actually implemented, mark it "[NOT IMPLEMENTED]" instead of
documenting it as if it exists. Do not carry over aspirational features
from the design doc into the documentation unless you find them in code.
Do not invent example values that weren't produced by an actual run —
pull real sample requests/responses from the test suite
(backend/tests/) or from actually calling the running localhost API.

OUTPUT FILES TO PRODUCE:
  docs/USER_GUIDE.md          — for someone using the web dashboard
  docs/API_REFERENCE.md       — for someone calling the REST API directly
  docs/DEVELOPER_GUIDE.md     — for someone reading/extending the code
  docs/DATA_AND_UPLOADS.md    — exactly where datasets/files/models go in/out

For EVERY section below, follow this exact template so the docs are
consistent and scannable:

  ### <Name>
  **Purpose:** one paragraph, what this does and why it exists.
  **Where it lives:** exact file path(s) in the repo.
  **How to access it:** URL (frontend) or method+path (API) or
    function signature (module).
  **Inputs / Parameters:** table of every parameter — name, type,
    required?, default, valid range/values, what it controls.
  **Example:** a REAL request/response or REAL usage snippet, pulled
    from a test or an actual call to the running server — not invented.
  **Output / Rendering:** what the user sees — which chart type, which
    table, which fields — or what JSON shape is returned, with field
    descriptions.
  **File/data upload details (if applicable):** exact endpoint or UI
    control used to upload, accepted file types/formats, size limits if
    enforced, where the file is stored after upload (disk path / DB
    table / object store), and what happens if the upload is malformed.
  **Errors:** the real error responses this can return (status codes +
    bodies), and what causes each.
  **Depends on / Feeds into:** which other modules or views this
    connects to, so a reader can trace data flow.

===========================================================================
PART A — FRONTEND: DOCUMENT EACH OF THE 7 DASHBOARD VIEWS
===========================================================================
For each view/page component in frontend/src/pages/, apply the template
above, specifically covering:
  1. Login / Auth view
  2. Dashboard (overview) view
  3. Clients view
  4. Training Rounds view
  5. Quantum Jobs view
  6. Model Registry view
  7. Explainability view

For EACH view additionally answer:
  - What API call(s) does it make on load, and on user interaction
    (button clicks, form submits)? Give the exact endpoint(s).
  - What chart(s) does it render (Chart.js types: line/bar/scatter/etc.),
    what data is fed into them, and what each axis represents.
  - What form fields or file inputs exist on this page? For any file
    input (e.g., dataset upload, CSV upload for a new client), state:
    accepted extensions, where the file goes (which endpoint, which
    backend storage location), and what validation happens before it's
    accepted.
  - What states does the view handle (loading, empty, error, success)
    and what does the user see in each?
  - Any real screenshots aren't required, but describe the layout
    (e.g., "top KPI cards, left sidebar filter, main chart, bottom
    paginated table") as it actually exists in the component code.

===========================================================================
PART B — BACKEND: DOCUMENT EVERY REST API ENDPOINT
===========================================================================
Read backend/api/routes_*.py and cross-reference with the live Swagger
docs at http://127.0.0.1:8000/docs. For EVERY endpoint apply the template,
plus:
  - Full path with version prefix (e.g., POST /api/v1/training/start)
  - Auth requirement: none / JWT required / role required
  - Request schema: every field, type, required/optional, validation
    rules (pull from the actual Pydantic model)
  - Response schema: every field, type, meaning
  - A REAL example curl command AND a real example response, generated
    by actually calling the running local server where safe to do so
    (GET/health endpoints), or pulled from backend/tests/ for endpoints
    that mutate state.
  - Rate limits or pagination behavior if implemented.

Organize this by router: auth, clients, training, quantum, predict/
explain, models, system.

Explicitly include a subsection: "File & Dataset Upload Endpoints" that
lists every endpoint accepting file/multipart data, with: accepted
formats (csv/json/parquet/etc.), max size if enforced, where the raw
file ends up (exact folder or table), and how it gets partitioned across
simulated clients (reference clients_simulation/data_partitioner.py).

===========================================================================
PART C — CORE MODULES: DOCUMENT EACH ENGINE
===========================================================================
For each of the following, read the actual module and document per the
template — treat these as "developer-facing," i.e., document the public
functions/classes a developer would call, not just endpoints:

  1. Federated Engine (backend/federated/) — server.py, client.py,
     strategies/fedavg.py, strategies/fedprox.py, round_manager.py.
     Include: how a round is started, how many rounds ran in your last
     benchmark, what FedAvg vs FedProx config differs.

  2. Quantum Optimization Engine (backend/quantum/) — qubo_builder.py,
     qaoa_optimizer.py, classical_fallback.py, feature_selection_qubo.py,
     client_selection_qubo.py, hyperparam_qubo.py, job_manager.py.
     Include for each: qubit count actually used, circuit depth actually
     used, simulator backend (Aer/PennyLane device name), and the
     real objective value + runtime from your last benchmark run — cite
     the source (test output or benchmark_runner.py results), don't
     restate the walkthrough numbers as if independently re-verified
     unless you actually re-run them.

  3. AutoML Engine (backend/automl/) — dataset_profiler.py,
     feature_selector.py, model_selector.py, hpo_classical.py,
     hpo_quantum_bridge.py, leaderboard.py.

  4. Models (backend/models/) — classical_models.py,
     transformer_model.py (document actual architecture used: layers,
     heads, hidden dim, as coded — not the design-doc defaults),
     model_registry.py (how versioning actually works).

  5. Privacy (backend/privacy/) — differential_privacy.py,
     secure_aggregation.py, privacy_budget_tracker.py. State the actual
     epsilon/delta values used in your benchmark run and where they're
     configured (env var / config file / API param).

  6. Security (backend/security/) — auth.py, tls_config.py,
     threat_detection.py, audit_log.py. Document exactly which threats
     from the original threat model are actually mitigated by code that
     exists (cross-check against test_security_privacy.py), and mark
     anything else as [NOT IMPLEMENTED] rather than implying coverage.

  7. Explainability (backend/explainability/) — shap_explainer.py,
     lime_explainer.py, attention_visualizer.py, report_generator.py.
     Include a real example: one input, its SHAP/LIME output, and the
     generated human-readable explanation text.

  8. Evaluation (backend/evaluation/) — metrics.py,
     federated_metrics.py, quantum_metrics.py, benchmark_runner.py.
     Reproduce the 4-baseline comparison table from your walkthrough
     here, but sourced directly from benchmark_runner.py's output
     format, and note the exact dataset/split used to produce it so the
     numbers are reproducible by someone else running the same command.

===========================================================================
PART D — DATA, FILES & STORAGE (docs/DATA_AND_UPLOADS.md)
===========================================================================
Answer explicitly, as a single reference table:
  - Where does a NEW dataset get uploaded (UI path + API endpoint)?
  - What format(s) are accepted, and what's the validation/rejection
    behavior for a bad file?
  - Where is it stored after upload (disk path, DB table + column,
    or object storage)?
  - How does it get split into per-client shards (function + params:
    IID vs Non-IID, number of clients, split ratios)?
  - Where do trained model artifacts get saved (path/table), and how
    are versions distinguished?
  - Where do quantum job results get persisted (quantum_jobs table —
    list its real columns as defined in database/models_orm.py)?
  - What gets logged for security/audit events, and where (table/file)?
  - How does a user download a trained model or an explanation report,
    if that's implemented — exact endpoint or UI button.

===========================================================================
PART E — DATABASE SCHEMA REFERENCE
===========================================================================
For every table in backend/database/models_orm.py, produce a table with:
column name | type | nullable | PK/FK | purpose. Pull this directly from
the ORM definitions and migrations, not from the original design doc, in
case the implemented schema diverged.

===========================================================================
PART F — CONFIGURATION REFERENCE
===========================================================================
List every environment variable / config setting read in
backend/config.py and frontend build config, with: name, purpose,
default value, and where it's set (.env.example).

===========================================================================
PART G — RUNNING IT (condensed, cross-link to setup_guide.md)
===========================================================================
One page: exact commands to start Postgres, backend, and frontend from a
clean checkout, and how to run the full 56-test suite and the
benchmark_runner.py comparison, so a new reader can reproduce the
walkthrough results themselves.

===========================================================================
FINAL STEP
===========================================================================
After generating all four docs, produce a short docs/INDEX.md linking to
each, and list any [NOT IMPLEMENTED] or [EXPERIMENTAL] items you found
during this documentation pass in one consolidated "Known Gaps" section
at the bottom of DEVELOPER_GUIDE.md, so the gap between the original
design doc and the actual shipped code is visible and honest.
```

---

## Why it's structured this way

- **Introspection over invention** — the agent is told repeatedly to read real files and pull real examples (from `backend/tests/` or a live localhost call), not restate the architecture doc as if it were the implementation. That's the main way doc-generation prompts go wrong: they regenerate the *design*, not the *build*.
- **Upload/storage is its own file (Part D)** — since that was your specific ask, it's pulled out as a standalone reference instead of buried inside each endpoint's section.
- **"Known Gaps" at the end** — keeps you honest for viva/interview purposes: anything in your original 33-section spec that didn't make it into code (e.g., full secure aggregation vs. just TLS+FedAvg) gets surfaced explicitly rather than silently implied as done.
