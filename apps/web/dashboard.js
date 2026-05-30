const state = {
  integrations: [],
};

const summaryGrid = document.getElementById("summaryGrid");
const projectsGrid = document.getElementById("projectsGrid");
const telemetryList = document.getElementById("telemetryList");
const documentSnapshotList = document.getElementById("documentSnapshotList");
const runsTable = document.getElementById("runsTable");
const connectionSelect = document.getElementById("connectionSelect");
const integrationForm = document.getElementById("integrationForm");
const collectForm = document.getElementById("collectForm");
const refreshButton = document.getElementById("refreshButton");
const toast = document.getElementById("toast");

function riskClass(level) {
  return `risk-chip risk-${level || "low"}`;
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  toast.style.background = isError ? "#8f1d1d" : "#15212b";
  window.setTimeout(() => toast.classList.add("hidden"), 2800);
}

function emptyState(message) {
  return `<div class="empty-state">${message}</div>`;
}

function renderSummary(summary) {
  const entries = [
    ["Integrations", summary.integrations_count],
    ["Telemetry snapshots", summary.telemetry_count],
    ["Document snapshots", summary.document_snapshot_count],
    ["Pipeline runs", summary.run_count],
    ["Active projects", summary.active_projects_count],
  ];
  summaryGrid.innerHTML = entries
    .map(
      ([label, value]) => `
        <article class="metric-card">
          <span class="label">${label}</span>
          <span class="value">${value}</span>
        </article>
      `,
    )
    .join("");
}

function renderProjects(projects) {
  if (!projects.length) {
    projectsGrid.innerHTML = emptyState("No telemetry projects yet. Create an integration and collect one snapshot.");
    return;
  }
  projectsGrid.innerHTML = projects
    .map(
      (project) => `
        <article class="project-card">
          <span class="meta-label">${project.provider}</span>
          <h3>${project.project_ref}</h3>
          <p>Latest score: <strong>${project.risk_score}/100</strong></p>
          <div class="${riskClass(project.risk_level)}">${project.risk_level}</div>
        </article>
      `,
    )
    .join("");
}

function renderTelemetry(snapshots) {
  if (!snapshots.length) {
    telemetryList.innerHTML = emptyState("No telemetry snapshots yet.");
    return;
  }
  telemetryList.innerHTML = snapshots
    .map((snapshot) => {
      const reasons = snapshot.evidence?.reasons || [];
      return `
        <article class="snapshot-card">
          <span class="meta-label">${snapshot.provider}</span>
          <h3>${snapshot.project_ref}</h3>
          <div class="${riskClass(snapshot.risk_level)}">${snapshot.risk_level} · ${snapshot.risk_score}/100</div>
          <div class="snapshot-meta">
            <span>${new Date(snapshot.created_at).toLocaleString()}</span>
            <span>${Object.entries(snapshot.metrics)
              .slice(0, 3)
              .map(([key, value]) => `${key}: ${value}`)
              .join(" · ")}</span>
          </div>
          ${reasons.length ? `<ul class="snapshot-reasons">${reasons.map((item) => `<li>${item}</li>`).join("")}</ul>` : ""}
        </article>
      `;
    })
    .join("");
}

function renderDocumentSnapshots(snapshots) {
  if (!snapshots.length) {
    documentSnapshotList.innerHTML = emptyState("No document-based project snapshots yet.");
    return;
  }
  documentSnapshotList.innerHTML = snapshots
    .map(
      (snapshot) => `
        <article class="snapshot-card">
          <span class="meta-label">run ${snapshot.run_id}</span>
          <h3>Document risk snapshot</h3>
          <div class="${riskClass(snapshot.risk_level)}">${snapshot.risk_level} · ${snapshot.risk_score}/100</div>
          <div class="snapshot-meta">
            <span>Actions: ${snapshot.actions_detected}</span>
            <span>Owners: ${snapshot.owners_detected}</span>
            <span>Blockers: ${snapshot.blockers_detected}</span>
            <span>Dependencies: ${snapshot.dependencies_detected}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderRuns(runs) {
  if (!runs.length) {
    runsTable.innerHTML = emptyState("No pipeline runs yet.");
    return;
  }
  runsTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Job</th>
          <th>Pipeline</th>
          <th>Status</th>
          <th>Step</th>
          <th>Started</th>
        </tr>
      </thead>
      <tbody>
        ${runs
          .map(
            (run) => `
              <tr>
                <td>${run.id}</td>
                <td>${run.job_id}</td>
                <td>${run.pipeline}</td>
                <td>${run.status}</td>
                <td>${run.current_step || "-"}</td>
                <td>${new Date(run.started_at).toLocaleString()}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderConnectionOptions() {
  if (!state.integrations.length) {
    connectionSelect.innerHTML = `<option value="">No connections yet</option>`;
    return;
  }
  connectionSelect.innerHTML = state.integrations
    .map((item) => `<option value="${item.id}">${item.name} · ${item.provider}</option>`)
    .join("");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function loadIntegrations() {
  state.integrations = await fetchJson("/integrations");
  renderConnectionOptions();
}

async function loadOverview() {
  const overview = await fetchJson("/dashboard/overview");
  renderSummary(overview.summary);
  renderProjects(overview.active_projects);
  renderTelemetry(overview.telemetry_snapshots);
  renderDocumentSnapshots(overview.document_snapshots);
  renderRuns(overview.recent_runs);
}

async function refreshAll() {
  try {
    await Promise.all([loadIntegrations(), loadOverview()]);
  } catch (error) {
    showToast(error.message, true);
  }
}

integrationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(integrationForm);
  const payload = Object.fromEntries(formData.entries());
  payload.config = {};
  try {
    await fetchJson("/integrations", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    integrationForm.reset();
    await refreshAll();
    showToast("Integration created");
  } catch (error) {
    showToast(error.message, true);
  }
});

collectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(collectForm);
  const payload = Object.fromEntries(formData.entries());
  try {
    await fetchJson("/telemetry/collect", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    collectForm.reset();
    renderConnectionOptions();
    await refreshAll();
    showToast("Telemetry snapshot collected");
  } catch (error) {
    showToast(error.message, true);
  }
});

refreshButton.addEventListener("click", () => {
  refreshAll();
});

refreshAll();
