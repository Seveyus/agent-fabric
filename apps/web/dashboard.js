const state = {
  integrations: [],
  recentProjects: [],
  selectedDiscoveryIntegrationId: "",
};

const summaryGrid = document.getElementById("summaryGrid");
const projectsGrid = document.getElementById("projectsGrid");
const telemetryList = document.getElementById("telemetryList");
const documentSnapshotList = document.getElementById("documentSnapshotList");
const runsTable = document.getElementById("runsTable");
const connectionSelect = document.getElementById("connectionSelect");
const integrationForm = document.getElementById("integrationForm");
const collectForm = document.getElementById("collectForm");
const discoveryForm = document.getElementById("discoveryForm");
const searchForm = document.getElementById("searchForm");
const forecastForm = document.getElementById("forecastForm");
const simulationForm = document.getElementById("simulationForm");
const worldStateForm = document.getElementById("worldStateForm");
const refreshButton = document.getElementById("refreshButton");
const toast = document.getElementById("toast");
const searchResults = document.getElementById("searchResults");
const forecastList = document.getElementById("forecastList");
const simulationList = document.getElementById("simulationList");
const worldStateList = document.getElementById("worldStateList");
const integrationList = document.getElementById("integrationList");
const syncRunList = document.getElementById("syncRunList");
const graphRelationList = document.getElementById("graphRelationList");
const discoveryConnectionSelect = document.getElementById("discoveryConnectionSelect");
const recentProjectList = document.getElementById("recentProjectList");
const syncRecentButton = document.getElementById("syncRecentButton");

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
    ["Metric snapshots", summary.telemetry_count],
    ["Document snapshots", summary.document_snapshot_count],
    ["Pipeline runs", summary.run_count],
    ["Active projects", summary.active_projects_count],
    ["Sync runs", summary.sync_runs_count],
    ["Forecasts", summary.forecasts_count],
    ["Simulations", summary.simulations_count],
    ["World states", summary.world_states_count],
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
    .map((project) => {
      const reasons = (project.top_reasons || []).map((item) => `<li>${item}</li>`).join("");
      const actions = (project.recommended_actions || []).map((item) => `<li>${item}</li>`).join("");
      const relations = (project.top_relationships || [])
        .map((item) => `<span class="relation-pill">${item.relation_type}: ${item.source_ref} -> ${item.target_ref}</span>`)
        .join("");
      return `
        <article class="project-card">
          <span class="meta-label">${project.provider}</span>
          <h3>${project.project_ref}</h3>
          <p>Unified score: <strong>${Math.round(project.risk_score * 100)}/100</strong></p>
          <div class="${riskClass(project.risk_level)}">${project.risk_level}</div>
          <div class="snapshot-meta">
            <span>Last sync: ${project.last_sync_status}</span>
            ${project.delay_probability !== null && project.delay_probability !== undefined ? `<span>Delay probability: ${Math.round(project.delay_probability * 100)}%</span>` : ""}
            ${project.forecast_status ? `<span>Forecast: ${project.forecast_status}</span>` : ""}
          </div>
          ${reasons ? `<div class="detail-block"><span class="meta-label">Top reasons</span><ul class="detail-list">${reasons}</ul></div>` : ""}
          ${actions ? `<div class="detail-block"><span class="meta-label">Recommended actions</span><ul class="detail-list">${actions}</ul></div>` : ""}
          ${relations ? `<div class="relation-wrap">${relations}</div>` : ""}
        </article>
      `;
    })
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
          ${snapshot.risk_level ? `<div class="${riskClass(snapshot.risk_level)}">${snapshot.risk_level} · ${snapshot.risk_score}/100</div>` : ""}
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

function renderSearchResults(results) {
  if (!results.length) {
    searchResults.innerHTML = emptyState("No search results yet.");
    return;
  }
  searchResults.innerHTML = results
    .map(
      (item) => `
        <article class="snapshot-card">
          <span class="meta-label">${item.result_type}</span>
          <h3>${item.title}</h3>
          <p>${item.summary}</p>
          <div class="snapshot-meta">
            <span>score: ${item.score}</span>
            <span>${item.ref}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderForecasts(forecasts) {
  if (!forecasts.length) {
    forecastList.innerHTML = emptyState("No forecasts yet.");
    return;
  }
  forecastList.innerHTML = forecasts
    .map(
      (item) => `
        <article class="snapshot-card">
          <span class="meta-label">${item.source}</span>
          <h3>${item.project_ref}</h3>
          <p>Delay probability: <strong>${Math.round(item.delay_probability * 100)}%</strong></p>
          <div class="snapshot-meta">
            <span>Trend: ${item.risk_trend}</span>
            <span>${item.forecast.projected_status}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderSimulations(simulations) {
  if (!simulations.length) {
    simulationList.innerHTML = emptyState("No simulations yet.");
    return;
  }
  simulationList.innerHTML = simulations
    .map(
      (item) => `
        <article class="snapshot-card">
          <span class="meta-label">${item.project_ref}</span>
          <h3>${item.scenario_name}</h3>
          <div class="${riskClass(item.outcome.risk_level)}">${item.outcome.risk_level} · ${item.outcome.risk_score}/100</div>
          <div class="snapshot-meta">
            <span>${JSON.stringify(item.adjustments)}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderWorldStates(states) {
  if (!states.length) {
    worldStateList.innerHTML = emptyState("No world states yet.");
    return;
  }
  worldStateList.innerHTML = states
    .map(
      (item) => `
        <article class="snapshot-card">
          <span class="meta-label">${item.state_kind}</span>
          <h3>${item.project_ref}</h3>
          <p>${item.transitions.interpretation}</p>
          <div class="snapshot-meta">
            <span>nodes: ${item.latent_state.node_count}</span>
            <span>edges: ${item.latent_state.edge_count}</span>
            <span>mean risk: ${item.latent_state.recent_risk_mean}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderIntegrations(integrations) {
  if (!integrations.length) {
    integrationList.innerHTML = emptyState("No integrations connected yet.");
    return;
  }
  integrationList.innerHTML = integrations
    .map(
      (item) => `
        <article class="snapshot-card">
          <span class="meta-label">${item.provider}</span>
          <h3>${item.name}</h3>
          <div class="snapshot-meta">
            <span>Status: ${item.status || "connected"}</span>
            <span>Created: ${new Date(item.created_at).toLocaleString()}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderSyncRuns(syncRuns) {
  if (!syncRuns.length) {
    syncRunList.innerHTML = emptyState("No sync runs yet.");
    return;
  }
  syncRunList.innerHTML = syncRuns
    .map(
      (item) => `
        <article class="snapshot-card">
          <span class="meta-label">${item.provider}</span>
          <h3>${item.project_ref}</h3>
          <div class="${riskClass(item.status === "failed" ? "high" : item.status === "completed" ? "low" : "medium")}">${item.status}</div>
          <div class="snapshot-meta">
            <span>raw: ${item.raw_count}</span>
            <span>entities: ${item.entity_count}</span>
            <span>relations: ${item.relation_count}</span>
            <span>metrics: ${item.snapshot_count}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderGraphRelations(relations) {
  if (!relations.length) {
    graphRelationList.innerHTML = emptyState("No graph relationships yet.");
    return;
  }
  graphRelationList.innerHTML = relations
    .map(
      (item) => `
        <article class="snapshot-card">
          <span class="meta-label">${item.project_ref || "cross-project"}</span>
          <h3>${item.relation_type}</h3>
          <div class="snapshot-meta">
            <span>${item.source_ref}</span>
            <span>-></span>
            <span>${item.target_ref}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderRecentProjects(projects) {
  if (!projects.length) {
    recentProjectList.innerHTML = emptyState("No recent projects discovered yet.");
    return;
  }
  recentProjectList.innerHTML = projects
    .map(
      (item) => `
        <article class="snapshot-card">
          <span class="meta-label">${item.provider}</span>
          <h3>${item.display_name}</h3>
          <div class="snapshot-meta">
            <span>${item.project_ref}</span>
            ${item.last_activity_at ? `<span>Last activity: ${new Date(item.last_activity_at).toLocaleString()}</span>` : ""}
          </div>
          <div class="action-row">
            <button class="button button-secondary button-small" type="button" data-action="use-project" data-project-ref="${item.project_ref}">Use in forms</button>
            <button class="button button-primary button-small" type="button" data-action="sync-project" data-project-ref="${item.project_ref}">Sync now</button>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderConnectionOptions() {
  if (!state.integrations.length) {
    connectionSelect.innerHTML = `<option value="">No connections yet</option>`;
    discoveryConnectionSelect.innerHTML = `<option value="">No connections yet</option>`;
    return;
  }
  const options = state.integrations.map((item) => `<option value="${item.id}">${item.name} · ${item.provider}</option>`).join("");
  connectionSelect.innerHTML = options;
  discoveryConnectionSelect.innerHTML = options;
  if (!state.selectedDiscoveryIntegrationId || !state.integrations.some((item) => item.id === state.selectedDiscoveryIntegrationId)) {
    state.selectedDiscoveryIntegrationId = state.integrations[0].id;
  }
  if (!connectionSelect.value) {
    connectionSelect.value = state.integrations[0].id;
  }
  discoveryConnectionSelect.value = state.selectedDiscoveryIntegrationId;
}

function fillProjectForms(projectRef) {
  collectForm.elements.project_ref.value = projectRef;
  forecastForm.elements.project_ref.value = projectRef;
  simulationForm.elements.project_ref.value = projectRef;
  worldStateForm.elements.project_ref.value = projectRef;
  searchForm.elements.project_ref.value = projectRef;
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

async function discoverRecentProjects() {
  const integrationId = discoveryConnectionSelect.value;
  const limit = discoveryForm.elements.limit.value || "6";
  if (!integrationId) {
    state.recentProjects = [];
    renderRecentProjects([]);
    return;
  }
  state.selectedDiscoveryIntegrationId = integrationId;
  state.recentProjects = await fetchJson(
    `/integrations/${encodeURIComponent(integrationId)}/recent-projects?limit=${encodeURIComponent(limit)}`,
  );
  renderRecentProjects(state.recentProjects);
}

async function loadOverview() {
  const overview = await fetchJson("/dashboard/overview");
  renderSummary(overview.summary);
  renderProjects(overview.active_projects);
  renderTelemetry(overview.telemetry_snapshots);
  renderDocumentSnapshots(overview.document_snapshots);
  renderRuns(overview.recent_runs);
  renderForecasts(overview.forecasts);
  renderSimulations(overview.simulations);
  renderWorldStates(overview.world_states);
  renderIntegrations(overview.integrations);
  renderSyncRuns(overview.sync_runs);
  renderGraphRelations(overview.graph_relationships);
}

async function refreshAll() {
  try {
    await Promise.all([loadIntegrations(), loadOverview()]);
    if (state.integrations.length) {
      await discoverRecentProjects();
    } else {
      renderRecentProjects([]);
    }
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

discoveryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await discoverRecentProjects();
    showToast("Recent projects discovered");
  } catch (error) {
    showToast(error.message, true);
  }
});

discoveryConnectionSelect.addEventListener("change", async () => {
  state.selectedDiscoveryIntegrationId = discoveryConnectionSelect.value;
  try {
    await discoverRecentProjects();
  } catch (error) {
    showToast(error.message, true);
  }
});

syncRecentButton.addEventListener("click", async () => {
  const integrationId = discoveryConnectionSelect.value;
  const limit = discoveryForm.elements.limit.value || "6";
  if (!integrationId) {
    showToast("Select an integration first", true);
    return;
  }
  try {
    const result = await fetchJson(
      `/integrations/${encodeURIComponent(integrationId)}/sync-recent?limit=${encodeURIComponent(limit)}`,
      { method: "POST" },
    );
    await refreshAll();
    showToast(`Synced ${result.synced_count}/${result.discovered_count} recent projects`);
  } catch (error) {
    showToast(error.message, true);
  }
});

recentProjectList.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) {
    return;
  }
  const projectRef = button.dataset.projectRef;
  if (button.dataset.action === "use-project") {
    fillProjectForms(projectRef);
    showToast(`Filled forms with ${projectRef}`);
    return;
  }
  if (button.dataset.action === "sync-project") {
    try {
      await fetchJson("/telemetry/collect", {
        method: "POST",
        body: JSON.stringify({
          integration_id: discoveryConnectionSelect.value,
          project_ref: projectRef,
        }),
      });
      fillProjectForms(projectRef);
      await refreshAll();
      showToast(`Synced ${projectRef}`);
    } catch (error) {
      showToast(error.message, true);
    }
  }
});

searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(searchForm);
  const payload = Object.fromEntries(formData.entries());
  if (!payload.project_ref) {
    delete payload.project_ref;
  }
  payload.limit = 10;
  try {
    const response = await fetchJson("/knowledge/search", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderSearchResults(response.results);
    showToast("Knowledge search complete");
  } catch (error) {
    showToast(error.message, true);
  }
});

forecastForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(forecastForm);
  const payload = Object.fromEntries(formData.entries());
  try {
    await fetchJson(`/knowledge/forecast/${encodeURIComponent(payload.project_ref)}`, {
      method: "POST",
    });
    await refreshAll();
    showToast("Forecast created");
  } catch (error) {
    showToast(error.message, true);
  }
});

simulationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(simulationForm);
  const payload = Object.fromEntries(formData.entries());
  try {
    const adjustments = JSON.parse(payload.adjustments_json);
    await fetchJson("/knowledge/simulate", {
      method: "POST",
      body: JSON.stringify({
        project_ref: payload.project_ref,
        scenario_name: payload.scenario_name,
        adjustments,
      }),
    });
    await refreshAll();
    showToast("Simulation complete");
  } catch (error) {
    showToast(error.message, true);
  }
});

worldStateForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(worldStateForm);
  const payload = Object.fromEntries(formData.entries());
  try {
    await fetchJson(`/knowledge/world-state/${encodeURIComponent(payload.project_ref)}`, {
      method: "POST",
    });
    await refreshAll();
    showToast("World state built");
  } catch (error) {
    showToast(error.message, true);
  }
});

refreshButton.addEventListener("click", () => {
  refreshAll();
});

refreshAll();
