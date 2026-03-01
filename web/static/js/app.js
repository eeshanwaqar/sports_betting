/**
 * EPL Predictor - Frontend Application
 *
 * Single-page application with client-side routing.
 * Communicates with the FastAPI backend via fetch().
 */

const API_BASE = window.EPL_API_BASE || "http://localhost:8000";  // injected by deploy workflow; falls back to localhost for dev

// ============================================================
// State
// ============================================================
const state = {
    teams: [],
    currentPage: "predict",
    modelInfo: null,
};

// ============================================================
// Initialization
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    checkApiHealth();
    loadTeams();
    loadModelInfo();
});

// ============================================================
// Navigation
// ============================================================
function initNavigation() {
    document.querySelectorAll(".nav-link").forEach((link) => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const page = link.dataset.page;
            navigateTo(page);
        });
    });
}

function navigateTo(page) {
    state.currentPage = page;

    // Update nav
    document.querySelectorAll(".nav-link").forEach((l) => l.classList.remove("active"));
    document.querySelector(`.nav-link[data-page="${page}"]`).classList.add("active");

    // Update pages
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    document.getElementById(`page-${page}`).classList.add("active");
}

// ============================================================
// API Helpers
// ============================================================
async function apiFetch(path, options = {}) {
    const resp = await fetch(`${API_BASE}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
}

async function checkApiHealth() {
    const dot = document.querySelector(".status-dot");
    const text = document.querySelector(".status-text");
    try {
        const data = await apiFetch("/health/ready");
        if (data.model_loaded) {
            dot.className = "status-dot connected";
            text.textContent = `Model: ${data.model_type}`;
        } else {
            dot.className = "status-dot";
            text.textContent = "Model not loaded";
        }
    } catch {
        dot.className = "status-dot error";
        text.textContent = "API offline";
    }
}

// ============================================================
// Load Teams (shared across pages)
// ============================================================
async function loadTeams() {
    try {
        const data = await apiFetch("/teams");
        state.teams = data.teams;
        populateTeamDropdowns();
        renderTeamsGrid();
    } catch (err) {
        showToast(`Failed to load teams: ${err.message}`, "error");
    }
}

function populateTeamDropdowns() {
    const selects = ["homeTeam", "awayTeam", "h2hTeamA", "h2hTeamB"];
    selects.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        const currentVal = el.value;
        el.innerHTML = '<option value="">Select team...</option>';
        state.teams.forEach((team) => {
            const opt = document.createElement("option");
            opt.value = team;
            opt.textContent = team;
            el.appendChild(opt);
        });
        if (currentVal) el.value = currentVal;
    });

    // Wire up selection handlers
    const homeEl = document.getElementById("homeTeam");
    const awayEl = document.getElementById("awayTeam");
    const predictBtn = document.getElementById("predictBtn");

    const updatePredictBtn = () => {
        predictBtn.disabled = !(homeEl.value && awayEl.value && homeEl.value !== awayEl.value);
    };
    homeEl.addEventListener("change", updatePredictBtn);
    awayEl.addEventListener("change", updatePredictBtn);
    predictBtn.addEventListener("click", handlePredict);

    // H2H
    const h2hA = document.getElementById("h2hTeamA");
    const h2hB = document.getElementById("h2hTeamB");
    const h2hBtn = document.getElementById("h2hBtn");

    const updateH2hBtn = () => {
        h2hBtn.disabled = !(h2hA.value && h2hB.value && h2hA.value !== h2hB.value);
    };
    h2hA.addEventListener("change", updateH2hBtn);
    h2hB.addEventListener("change", updateH2hBtn);
    h2hBtn.addEventListener("click", handleH2H);
}

// ============================================================
// Predict Page
// ============================================================
async function handlePredict() {
    const homeTeam = document.getElementById("homeTeam").value;
    const awayTeam = document.getElementById("awayTeam").value;
    const btn = document.getElementById("predictBtn");
    const btnText = btn.querySelector(".btn-text");
    const btnLoader = btn.querySelector(".btn-loader");
    const results = document.getElementById("predictionResults");
    const errorEl = document.getElementById("predictError");

    // Loading state
    btn.disabled = true;
    btnText.textContent = "Predicting...";
    btnLoader.classList.remove("hidden");
    results.classList.add("hidden");
    errorEl.classList.add("hidden");

    try {
        const data = await apiFetch("/predict", {
            method: "POST",
            body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
        });
        renderPrediction(data);
        results.classList.remove("hidden");
    } catch (err) {
        errorEl.textContent = err.message;
        errorEl.classList.remove("hidden");
    } finally {
        btn.disabled = false;
        btnText.textContent = "Get Prediction";
        btnLoader.classList.add("hidden");
    }
}

function renderPrediction(data) {
    const p = data.probabilities;
    const o = data.odds;

    // Header
    document.getElementById("resultMatchup").textContent =
        `${data.home_team} vs ${data.away_team}`;

    const outcomeEl = document.getElementById("resultOutcome");
    const outcomeMap = { H: "Home Win", D: "Draw", A: "Away Win" };
    const classMap = { H: "home", D: "draw", A: "away" };
    outcomeEl.className = `result-outcome ${classMap[data.prediction]}`;
    outcomeEl.textContent = outcomeMap[data.prediction];

    // Probabilities — animate bars
    setTimeout(() => {
        setBar("probBarHome", "probHome", p.home_win);
        setBar("probBarDraw", "probDraw", p.draw);
        setBar("probBarAway", "probAway", p.away_win);
    }, 50);

    // Odds
    document.getElementById("oddsHome").textContent = o.home_win.toFixed(2);
    document.getElementById("oddsDraw").textContent = o.draw.toFixed(2);
    document.getElementById("oddsAway").textContent = o.away_win.toFixed(2);

    // Confidence
    const conf = data.confidence;
    setTimeout(() => {
        document.getElementById("confidenceBar").style.width = `${conf * 100}%`;
        document.getElementById("confidenceValue").textContent = `${(conf * 100).toFixed(1)}%`;
    }, 50);
}

function setBar(barId, valueId, prob) {
    document.getElementById(barId).style.width = `${prob * 100}%`;
    document.getElementById(valueId).textContent = `${(prob * 100).toFixed(1)}%`;
}

// ============================================================
// Teams Page
// ============================================================
function renderTeamsGrid() {
    const grid = document.getElementById("teamsGrid");
    if (!state.teams.length) {
        grid.innerHTML = '<div class="loading-spinner">No teams loaded</div>';
        return;
    }

    grid.innerHTML = state.teams
        .map(
            (team) => `
        <div class="team-card" data-team="${team}">
            <div class="team-card-name">${team}</div>
            <div class="team-card-meta">Click for details</div>
        </div>`
        )
        .join("");

    // Click handlers
    grid.querySelectorAll(".team-card").forEach((card) => {
        card.addEventListener("click", () => openTeamDetail(card.dataset.team));
    });

    // Search filter
    document.getElementById("teamSearch").addEventListener("input", (e) => {
        const q = e.target.value.toLowerCase();
        grid.querySelectorAll(".team-card").forEach((card) => {
            const match = card.dataset.team.toLowerCase().includes(q);
            card.style.display = match ? "" : "none";
        });
    });
}

async function openTeamDetail(teamName) {
    const modal = document.getElementById("teamModal");
    const detail = document.getElementById("teamDetail");

    detail.innerHTML = '<div class="loading-spinner">Loading...</div>';
    modal.classList.remove("hidden");

    // Close handlers
    modal.querySelector(".modal-overlay").onclick = () => modal.classList.add("hidden");
    modal.querySelector(".modal-close").onclick = () => modal.classList.add("hidden");

    try {
        const data = await apiFetch(`/teams/${encodeURIComponent(teamName)}`);
        detail.innerHTML = renderTeamDetail(data);
    } catch (err) {
        detail.innerHTML = `<p class="error-banner">${err.message}</p>`;
    }
}

function renderTeamDetail(team) {
    const formDots = team.recent_form
        ? team.recent_form
              .split("")
              .map((c) => `<div class="form-dot ${c}">${c}</div>`)
              .join("")
        : "<span style='color:var(--text-muted)'>No form data</span>";

    const hr = team.home_record;
    const ar = team.away_record;

    return `
        <div class="team-detail-header">
            <h2>${team.name}</h2>
            <p style="color:var(--text-secondary);font-size:0.9rem">${team.matches_played} matches played</p>
            <div style="margin-top:10px">
                <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;font-weight:600">Recent Form</span>
                <div class="team-detail-form">${formDots}</div>
            </div>
        </div>
        <div class="team-record-grid">
            <div class="record-card">
                <h4>Home Record</h4>
                <div class="record-stat"><span>Wins</span><span style="color:var(--accent-green)">${hr.wins}</span></div>
                <div class="record-stat"><span>Draws</span><span style="color:var(--accent-yellow)">${hr.draws}</span></div>
                <div class="record-stat"><span>Losses</span><span style="color:var(--accent-red)">${hr.losses}</span></div>
            </div>
            <div class="record-card">
                <h4>Away Record</h4>
                <div class="record-stat"><span>Wins</span><span style="color:var(--accent-green)">${ar.wins}</span></div>
                <div class="record-stat"><span>Draws</span><span style="color:var(--accent-yellow)">${ar.draws}</span></div>
                <div class="record-stat"><span>Losses</span><span style="color:var(--accent-red)">${ar.losses}</span></div>
            </div>
        </div>`;
}

// ============================================================
// Head-to-Head Page
// ============================================================
async function handleH2H() {
    const teamA = document.getElementById("h2hTeamA").value;
    const teamB = document.getElementById("h2hTeamB").value;
    const results = document.getElementById("h2hResults");
    const errorEl = document.getElementById("h2hError");

    results.classList.add("hidden");
    errorEl.classList.add("hidden");

    try {
        const data = await apiFetch(
            `/matches/head-to-head?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}`
        );
        renderH2H(data, teamA, teamB);
        results.classList.remove("hidden");
    } catch (err) {
        errorEl.textContent = err.message;
        errorEl.classList.remove("hidden");
    }
}

function renderH2H(data, teamA, teamB) {
    const s = data.summary;
    const winsA = s[`${teamA}_wins`] || 0;
    const winsB = s[`${teamB}_wins`] || 0;
    const draws = s.draws || 0;

    document.getElementById("h2hSummary").innerHTML = `
        <div class="h2h-team-stat">
            <div class="team-name">${teamA}</div>
            <div class="win-count">${winsA}</div>
            <div class="win-label">Wins</div>
        </div>
        <div class="h2h-draw-stat">
            <div class="draw-count">${draws}</div>
            <div class="draw-label">Draws</div>
            <div style="margin-top:8px;font-size:0.8rem;color:var(--text-muted)">${data.total_meetings} meetings</div>
        </div>
        <div class="h2h-team-stat">
            <div class="team-name">${teamB}</div>
            <div class="win-count">${winsB}</div>
            <div class="win-label">Wins</div>
        </div>`;

    const matchesHtml = data.recent_matches
        .map((m) => {
            return `
            <div class="match-row">
                <span class="match-date">${m.date}</span>
                <span class="match-team home">${m.home_team}</span>
                <span class="match-score">${m.home_goals} - ${m.away_goals}</span>
                <span class="match-team">${m.away_team}</span>
            </div>`;
        })
        .join("");

    document.getElementById("h2hMatches").innerHTML = `
        <h3>Recent Meetings</h3>
        ${matchesHtml || '<p style="color:var(--text-muted)">No meetings found</p>'}`;
}

// ============================================================
// Model Info Page
// ============================================================
async function loadModelInfo() {
    const container = document.getElementById("modelInfo");
    try {
        const [health, recent] = await Promise.all([
            apiFetch("/health/ready"),
            apiFetch("/matches/recent?limit=5"),
        ]);

        // Try to load model_info.json via a special endpoint or use health data
        state.modelInfo = health;
        renderModelInfo(health, recent);
    } catch (err) {
        container.innerHTML = `
            <div class="info-card" style="grid-column:1/-1">
                <h3>Status</h3>
                <p style="color:var(--accent-red);margin-top:0.5rem">
                    Could not load model info. Make sure the API is running and a trained model exists.
                </p>
                <p style="color:var(--text-muted);margin-top:0.5rem;font-size:0.85rem">${err.message}</p>
            </div>`;
    }
}

function renderModelInfo(health, recent) {
    const container = document.getElementById("modelInfo");

    // Status card
    const statusColor = health.model_loaded ? "var(--accent-green)" : "var(--accent-red)";
    const statusText = health.model_loaded ? "Ready" : "Not Loaded";

    // Recent matches
    const recentHtml = recent.matches
        .map(
            (m) => `
        <div class="match-row">
            <span class="match-date">${m.date}</span>
            <span class="match-team home">${m.home_team}</span>
            <span class="match-score">${m.home_goals} - ${m.away_goals}</span>
            <span class="match-team">${m.away_team}</span>
        </div>`
        )
        .join("");

    container.innerHTML = `
        <div class="info-card">
            <h3>System Status</h3>
            <div class="info-stat-row">
                <span class="info-stat-label">API Status</span>
                <span class="info-stat-value" style="color:${statusColor}">${statusText}</span>
            </div>
            <div class="info-stat-row">
                <span class="info-stat-label">Model Type</span>
                <span class="info-stat-value">${health.model_type || "N/A"}</span>
            </div>
            <div class="info-stat-row">
                <span class="info-stat-label">API Version</span>
                <span class="info-stat-value">${health.version}</span>
            </div>
            <div class="info-stat-row">
                <span class="info-stat-label">Teams Available</span>
                <span class="info-stat-value">${state.teams.length}</span>
            </div>
        </div>

        <div class="info-card">
            <h3>Quick Links</h3>
            <div class="info-stat-row">
                <span class="info-stat-label">API Documentation</span>
                <a href="${API_BASE}/docs" target="_blank" class="info-stat-value" style="color:var(--accent-blue-hover);text-decoration:none">
                    /docs
                </a>
            </div>
            <div class="info-stat-row">
                <span class="info-stat-label">ReDoc</span>
                <a href="${API_BASE}/redoc" target="_blank" class="info-stat-value" style="color:var(--accent-blue-hover);text-decoration:none">
                    /redoc
                </a>
            </div>
            <div class="info-stat-row">
                <span class="info-stat-label">Health Check</span>
                <a href="${API_BASE}/health/ready" target="_blank" class="info-stat-value" style="color:var(--accent-blue-hover);text-decoration:none">
                    /health/ready
                </a>
            </div>
            <div class="info-stat-row">
                <span class="info-stat-label">MLflow UI</span>
                <span class="info-stat-value" style="color:var(--text-muted)">localhost:5000</span>
            </div>
        </div>

        <div class="info-card" style="grid-column:1/-1">
            <h3>Recent Matches in Dataset</h3>
            ${recentHtml}
        </div>`;
}

// ============================================================
// Toast Notifications
// ============================================================
function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}
