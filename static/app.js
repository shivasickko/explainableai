/**
 * Explainable Diabetes Risk Prediction Dashboard - Client Application
 * Department of Artificial Intelligence and Data Science - Batch 10
 */

// Global State
let currentPatientData = {};
let samplePatients = [];
let modelMetrics = {};
let shapChart = null;
let whatIfBaselineData = {};

document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupPatientForm();
  await loadInitialData();
  
  // Run initial prediction with pre-filled default form values
  submitPredictionForm();
});

/* ==========================================================================
   1. Tab Navigation
   ========================================================================== */
function setupTabs() {
  const tabButtons = document.querySelectorAll('.nav-tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');

      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => {
        p.classList.remove('active');
        p.style.display = 'none';
      });

      btn.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
        targetPane.style.display = 'block';
      }

      // If switching to What-If, auto-sync with current patient if needed
      if (targetId === 'whatif-tab' && Object.keys(whatIfBaselineData).length === 0) {
        syncWhatIfWithCurrentPatient();
      }
    });
  });
}

/* ==========================================================================
   2. Data Loading & Initialization
   ========================================================================== */
async function loadInitialData() {
  try {
    // Fetch Sample Patients
    const spRes = await fetch('/api/sample-patients');
    if (spRes.ok) {
      samplePatients = await spRes.json();
      renderArchetypeButtons();
    }

    // Fetch Model Statistics
    const msRes = await fetch('/api/model-stats');
    if (msRes.ok) {
      modelMetrics = await msRes.json();
      renderModelStats();
    }
  } catch (err) {
    console.warn('Error fetching initial model data:', err);
  }
}

function renderArchetypeButtons() {
  const container = document.getElementById('archetypes-container');
  if (!container || !samplePatients.length) return;

  container.innerHTML = samplePatients.map(sp => `
    <button class="archetype-btn" type="button" onclick="loadSamplePatient('${sp.id}')">
      ${sp.title}
    </button>
  `).join('');
}

function renderModelStats() {
  if (!modelMetrics.metrics) return;

  const m = modelMetrics.metrics;
  const accEl = document.getElementById('stat-accuracy');
  const aucEl = document.getElementById('stat-auc');
  const recEl = document.getElementById('stat-recall');
  const specEl = document.getElementById('stat-specificity');
  const f1El = document.getElementById('stat-f1');

  if (accEl) accEl.innerText = `${(m.accuracy * 100).toFixed(1)}%`;
  if (aucEl) aucEl.innerText = m.roc_auc.toFixed(3);
  if (recEl) recEl.innerText = `${(m.recall_sensitivity * 100).toFixed(1)}%`;
  if (specEl) specEl.innerText = `${(m.specificity * 100).toFixed(1)}%`;
  if (f1El) f1El.innerText = m.f1_score.toFixed(3);

  // Confusion Matrix
  if (m.confusion_matrix) {
    const cm = m.confusion_matrix;
    const tn = document.getElementById('cm-tn');
    const fp = document.getElementById('cm-fp');
    const fn = document.getElementById('cm-fn');
    const tp = document.getElementById('cm-tp');
    if (tn) tn.innerText = `${cm.true_negative} (TN)`;
    if (fp) fp.innerText = `${cm.false_positive} (FP)`;
    if (fn) fn.innerText = `${cm.false_negative} (FN)`;
    if (tp) tp.innerText = `${cm.true_positive} (TP)`;
  }

  // Global Feature Importance
  const giContainer = document.getElementById('global-importance-list');
  if (giContainer && modelMetrics.feature_importance) {
    const maxImp = Math.max(...modelMetrics.feature_importance.map(f => f.importance));
    giContainer.innerHTML = modelMetrics.feature_importance.map((f, idx) => {
      const pct = (f.importance / maxImp) * 100;
      return `
        <div style="font-size: 0.825rem;">
          <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
            <span style="font-weight: 600; color: var(--neutral-800);">#${idx + 1} ${f.display_name}</span>
            <span style="font-family: var(--font-mono); color: var(--neutral-600);">${(f.importance * 100).toFixed(1)}%</span>
          </div>
          <div style="height: 8px; background: var(--neutral-100); border-radius: 4px; overflow: hidden;">
            <div style="height: 100%; width: ${pct}%; background: linear-gradient(90deg, #0284c7, #2563eb); border-radius: 4px;"></div>
          </div>
        </div>
      `;
    }).join('');
  }
}

/* ==========================================================================
   3. Patient Form & Prediction Execution
   ========================================================================== */
function setupPatientForm() {
  const form = document.getElementById('patient-form');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    submitPredictionForm();
  });
}

function getFormData() {
  const form = document.getElementById('patient-form');
  const formData = new FormData(form);
  const data = {};
  for (let [key, val] of formData.entries()) {
    data[key] = parseFloat(val);
  }
  return data;
}

async function submitPredictionForm() {
  const payload = getFormData();
  currentPatientData = payload;

  const btn = document.getElementById('btn-analyze');
  if (btn) {
    btn.innerHTML = `<span>⏳</span> Computing Shapley Attributions...`;
    btn.disabled = true;
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`Prediction API responded with status ${res.status}`);
    }

    const result = await res.json();
    renderPredictionResults(result);

  } catch (err) {
    console.error('Error in prediction:', err);
    alert('Failed to connect to prediction engine: ' + err.message);
  } finally {
    if (btn) {
      btn.innerHTML = `<span>🔍</span> Analyze Diabetes Risk & Feature Contributions`;
      btn.disabled = false;
    }
  }
}

function loadSamplePatient(archetypeId) {
  const archetype = samplePatients.find(p => p.id === archetypeId);
  if (!archetype) return;

  // Highlight button
  document.querySelectorAll('.archetype-btn').forEach(btn => {
    btn.classList.toggle('active', btn.textContent.includes(archetype.title.split(' ')[0]));
  });

  const d = archetype.data;
  document.getElementById('inp-glucose').value = d.Glucose;
  document.getElementById('inp-bmi').value = d.BMI;
  document.getElementById('inp-bp').value = d.BloodPressure;
  document.getElementById('inp-age').value = d.Age;
  document.getElementById('inp-insulin').value = d.Insulin;
  document.getElementById('inp-skinthickness').value = d.SkinThickness;
  document.getElementById('inp-pedigree').value = d.DiabetesPedigreeFunction;
  document.getElementById('inp-pregnancies').value = d.Pregnancies;

  // Auto trigger prediction
  submitPredictionForm();
}

function resetForm() {
  document.querySelectorAll('.archetype-btn').forEach(b => b.classList.remove('active'));
  loadSamplePatient('healthy_patient');
}

/* ==========================================================================
   4. Render Prediction & SHAP Feature Contribution Results
   ========================================================================== */
function renderPredictionResults(res) {
  const pred = res.prediction;
  const prob = pred.probability;
  const pct = pred.probability_percent;
  const tier = pred.risk_tier;
  const severity = pred.risk_severity;

  // 1. Update Dial and Banner
  const banner = document.getElementById('risk-banner');
  if (banner) {
    banner.className = `result-banner ${severity}`;
  }

  const badge = document.getElementById('risk-badge');
  if (badge) {
    badge.className = `risk-tier-badge ${severity}`;
    badge.innerText = tier;
  }

  const valEl = document.getElementById('risk-pct-val');
  if (valEl) valEl.innerText = `${pct}%`;

  const summaryEl = document.getElementById('clinical-summary-text');
  if (summaryEl) summaryEl.innerText = pred.clinical_summary;

  // SVG Circular Gauge
  const circle = document.getElementById('risk-dial-circle');
  if (circle) {
    const totalCircumference = 2 * Math.PI * 40; // ~251.2
    const offset = totalCircumference - (prob * totalCircumference);
    circle.style.strokeDashoffset = offset;
    circle.style.stroke = pred.risk_color;
  }

  // Baseline Comparison
  const baseEl = document.getElementById('baseline-compare-text');
  if (baseEl) {
    const diff = (pct - pred.base_probability_percent).toFixed(1);
    const sign = diff >= 0 ? '+' : '';
    baseEl.innerHTML = `Population Baseline: <strong>${pred.base_probability_percent}%</strong> (Shift: <strong>${sign}${diff}%</strong>)`;
  }

  // Top Factor Chips
  const chipsContainer = document.getElementById('top-factors-chips');
  if (chipsContainer) {
    let chipsHtml = '';
    if (res.top_risk_factors && res.top_risk_factors.length > 0) {
      const topRisk = res.top_risk_factors[0];
      chipsHtml += `<span class="pill" style="background: #fee2e2; color: #991b1b; border: 1px solid #fecaca;">
        🚨 Primary Risk Driver: ${topRisk.display_name} (+${topRisk.shap_value.toFixed(2)})
      </span>`;
    }
    if (res.top_protective_factors && res.top_protective_factors.length > 0) {
      const topProt = res.top_protective_factors[0];
      chipsHtml += `<span class="pill" style="background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0;">
        🛡️ Protective Factor: ${topProt.display_name} (${topProt.shap_value.toFixed(2)})
      </span>`;
    }
    chipsContainer.innerHTML = chipsHtml;
  }

  // 2. Render Waterfall / Bar Chart
  renderShapWaterfallChart(res.feature_contributions);

  // 3. Render Detailed Clinical Insights Table
  renderClinicalTable(res.feature_contributions);
}

function renderShapWaterfallChart(contributions) {
  // Sort for horizontal bar chart
  const sorted = [...contributions].reverse();
  const labels = sorted.map(c => c.display_name);
  const values = sorted.map(c => c.shap_value);
  const bgColors = sorted.map(c => c.shap_value >= 0 ? '#ef4444' : '#10b981');

  const canvas = document.getElementById('shap-waterfall-chart');
  const fallback = document.getElementById('waterfall-fallback');

  if (typeof Chart !== 'undefined' && canvas) {
    if (shapChart) {
      shapChart.destroy();
    }

    const ctx = canvas.getContext('2d');
    shapChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Shapley Contribution to Risk',
          data: values,
          backgroundColor: bgColors,
          borderRadius: 4,
          borderSkipped: false
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const val = context.raw;
                const sign = val >= 0 ? '+' : '';
                return ` SHAP Impact: ${sign}${val.toFixed(4)} (${val >= 0 ? 'Increases Risk' : 'Protective Factor'})`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: '#f1f5f9' },
            title: {
              display: true,
              text: '← Protective (Lowers Risk)  |  Elevates Diabetes Risk →',
              color: '#64748b',
              font: { size: 11, weight: 'bold' }
            }
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 12, weight: '500' } }
          }
        }
      }
    });

    if (fallback) fallback.style.display = 'none';

  } else if (fallback) {
    // Pure CSS/HTML Fallback in case Chart.js CDN is unavailable
    if (canvas) canvas.style.display = 'none';
    fallback.style.display = 'flex';

    const maxVal = Math.max(...contributions.map(c => Math.abs(c.shap_value)), 1.0);
    fallback.innerHTML = contributions.map(c => {
      const isRisk = c.shap_value >= 0;
      const widthPct = Math.min((Math.abs(c.shap_value) / maxVal) * 50, 50);
      return `
        <div class="waterfall-row">
          <div>
            <div class="waterfall-feat-name">${c.display_name}</div>
            <div class="waterfall-feat-val">${c.patient_value} ${c.unit}</div>
          </div>
          <div class="waterfall-bar-track">
            <div class="waterfall-center-line"></div>
            <div class="waterfall-bar-fill ${isRisk ? 'risk' : 'protective'}" style="width: ${widthPct}%;"></div>
          </div>
          <div class="waterfall-score ${isRisk ? 'score-risk' : 'score-protective'}">
            ${c.shap_value >= 0 ? '+' : ''}${c.shap_value.toFixed(2)}
          </div>
        </div>
      `;
    }).join('');
  }
}

function renderClinicalTable(contributions) {
  const tbody = document.getElementById('clinical-insights-tbody');
  if (!tbody) return;

  tbody.innerHTML = contributions.map(c => {
    let badgeClass = 'status-optimal';
    if (c.clinical_status.includes('High') || c.clinical_status.includes('Diabetic') || c.clinical_status.includes('Obese')) {
      badgeClass = 'status-high';
    } else if (c.clinical_status.includes('Elevated') || c.clinical_status.includes('Overweight')) {
      badgeClass = 'status-elevated';
    }

    const sign = c.shap_value >= 0 ? '+' : '';
    const shapColor = c.shap_value >= 0 ? 'var(--danger)' : 'var(--success)';

    return `
      <tr>
        <td>
          <div style="font-weight: 700; color: var(--neutral-900);">${c.display_name}</div>
          <div style="font-size: 0.75rem; color: var(--neutral-500);">${c.category}</div>
        </td>
        <td>
          <span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.95rem;">${c.patient_value}</span>
          <span style="font-size: 0.75rem; color: var(--neutral-500);">${c.unit}</span>
        </td>
        <td style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--neutral-600);">${c.normal_range_str}</td>
        <td>
          <span class="status-badge ${badgeClass}">${c.clinical_status}</span>
        </td>
        <td>
          <span style="font-family: var(--font-mono); font-weight: 700; color: ${shapColor};">
            ${sign}${c.shap_value.toFixed(3)}
          </span>
          <div style="font-size: 0.7rem; color: var(--neutral-500);">${c.direction}</div>
        </td>
        <td>
          <div style="font-size: 0.825rem; color: var(--neutral-800); margin-bottom: 0.25rem;">
            ${c.interpretation}
          </div>
          <div style="font-size: 0.775rem; color: #0284c7; font-weight: 500;">
            👉 <em>${c.action_recommendation}</em>
          </div>
        </td>
      </tr>
    `;
  }).join('');
}

/* ==========================================================================
   5. What-If Intervention Simulator
   ========================================================================== */
function syncWhatIfWithCurrentPatient() {
  whatIfBaselineData = { ...currentPatientData };
  if (!whatIfBaselineData.Glucose) {
    whatIfBaselineData = getFormData();
  }

  // Pre-fill slider positions
  const gSlider = document.getElementById('slider-glucose');
  const bSlider = document.getElementById('slider-bmi');
  const bpSlider = document.getElementById('slider-bp');
  const iSlider = document.getElementById('slider-insulin');

  if (gSlider) gSlider.value = whatIfBaselineData.Glucose || 110;
  if (bSlider) bSlider.value = whatIfBaselineData.BMI || 24.5;
  if (bpSlider) bpSlider.value = whatIfBaselineData.BloodPressure || 72;
  if (iSlider) iSlider.value = whatIfBaselineData.Insulin || 75;

  onWhatIfSliderChange();
}

let whatIfDebounce = null;
function onWhatIfSliderChange() {
  const gVal = parseFloat(document.getElementById('slider-glucose').value);
  const bVal = parseFloat(document.getElementById('slider-bmi').value);
  const bpVal = parseFloat(document.getElementById('slider-bp').value);
  const iVal = parseFloat(document.getElementById('slider-insulin').value);

  document.getElementById('val-target-glucose').innerText = `${gVal} mg/dL`;
  document.getElementById('val-target-bmi').innerText = `${bVal} kg/m²`;
  document.getElementById('val-target-bp').innerText = `${bpVal} mmHg`;
  document.getElementById('val-target-insulin').innerText = `${iVal} μU/mL`;

  clearTimeout(whatIfDebounce);
  whatIfDebounce = setTimeout(() => {
    executeWhatIfCalculation(gVal, bVal, bpVal, iVal);
  }, 120);
}

async function executeWhatIfCalculation(glucose, bmi, bp, insulin) {
  if (Object.keys(whatIfBaselineData).length === 0) {
    whatIfBaselineData = getFormData();
  }

  const modifiedData = {
    ...whatIfBaselineData,
    Glucose: glucose,
    BMI: bmi,
    BloodPressure: bp,
    Insulin: insulin
  };

  try {
    const res = await fetch('/api/what-if', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        baseline: whatIfBaselineData,
        modified: modifiedData
      })
    });

    if (!res.ok) throw new Error('What-If simulation failed');

    const result = await res.json();
    renderWhatIfResults(result);

  } catch (err) {
    console.error('Error running What-If scenario:', err);
  }
}

function renderWhatIfResults(result) {
  const deltaText = document.getElementById('whatif-delta-text');
  const statusPill = document.getElementById('whatif-status-pill');

  const basePct = result.baseline.probability_percent;
  const modPct = result.target.probability_percent;
  const redPct = result.percentage_risk_reduction;

  if (deltaText) {
    if (result.is_improved) {
      deltaText.innerText = `Risk Drops from ${basePct}% → ${modPct}% (${redPct}% Relative Reduction)`;
    } else {
      deltaText.innerText = `Risk Shifts from ${basePct}% → ${modPct}% (+${(modPct - basePct).toFixed(1)}% Change)`;
    }
  }

  if (statusPill) {
    if (result.is_improved && redPct > 20) {
      statusPill.className = 'pill pill-success';
      statusPill.innerText = `✅ Major Clinical Improvement (${redPct}% Lower Risk)`;
    } else if (result.is_improved) {
      statusPill.className = 'pill pill-primary';
      statusPill.innerText = `👍 Moderate Risk Reduction`;
    } else {
      statusPill.className = 'pill pill-neutral';
      statusPill.innerText = `⚠️ Elevated Risk Profile`;
    }
  }

  // Shifts Table
  const tbody = document.getElementById('whatif-shifts-tbody');
  if (tbody && result.feature_shifts) {
    tbody.innerHTML = result.feature_shifts.map(s => `
      <tr>
        <td style="font-weight: 600;">${s.display_name}</td>
        <td>${s.baseline_value} ${s.unit}</td>
        <td style="color: var(--primary); font-weight: 700;">${s.target_value} ${s.unit}</td>
        <td>
          <span style="font-family: var(--font-mono); font-weight: 600;">
            ${s.value_change > 0 ? '+' : ''}${s.value_change} ${s.unit}
          </span>
        </td>
        <td>
          <span class="pill ${s.risk_impact === 'Risk Reduced' ? 'pill-success' : 'pill-neutral'}">
            ${s.risk_impact} (${s.shap_delta > 0 ? '+' : ''}${s.shap_delta.toFixed(2)})
          </span>
        </td>
      </tr>
    `).join('');
  }
}

/* ==========================================================================
   6. Cohort Batch Screening
   ========================================================================== */
async function loadSampleBatchFile() {
  try {
    const res = await fetch('/static/sample_batch_patients.csv');
    let blob;
    if (res.ok) {
      blob = await res.blob();
    } else {
      // Fallback: fetch root sample_batch_patients.csv
      const rootRes = await fetch('/sample_batch_patients.csv');
      blob = await rootRes.blob();
    }

    const file = new File([blob], 'sample_batch_patients.csv', { type: 'text/csv' });
    uploadBatchFile(file);

  } catch (err) {
    console.error('Error fetching sample batch:', err);
    alert('Failed to load sample batch CSV: ' + err.message);
  }
}

function handleBatchFileUpload(event) {
  const file = event.target.files[0];
  if (file) {
    uploadBatchFile(file);
  }
}

async function uploadBatchFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const container = document.getElementById('batch-table-container');
  const statsGrid = document.getElementById('batch-stats-grid');
  const tbody = document.getElementById('batch-results-tbody');

  if (tbody) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 2rem;">Analyzing cohort dataset through TreeSHAP engine...</td></tr>`;
    if (container) container.style.display = 'block';
  }

  try {
    const res = await fetch('/api/batch-predict', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || 'Batch analysis failed');
    }

    const data = await res.json();

    // Stats
    if (statsGrid) {
      statsGrid.style.display = 'grid';
      document.getElementById('batch-total-screened').innerText = data.total_analyzed;
      document.getElementById('batch-low-count').innerText = data.risk_breakdown['Low Risk'] || 0;
      document.getElementById('batch-mod-count').innerText = data.risk_breakdown['Moderate Risk'] || 0;
      document.getElementById('batch-high-count').innerText = data.risk_breakdown['High Risk'] || 0;
    }

    // Rows
    if (tbody && data.patients) {
      tbody.innerHTML = data.patients.map((p, idx) => {
        let badgeColor = 'pill-success';
        if (p.risk_tier === 'Moderate Risk') badgeColor = 'pill-primary';
        if (p.risk_tier === 'High Risk') badgeColor = 'pill-neutral' ;

        return `
          <tr>
            <td style="font-weight: 700; font-family: var(--font-mono);">${p.patient_id}</td>
            <td>${p.glucose.toFixed(1)}</td>
            <td>${p.bmi.toFixed(1)}</td>
            <td>${p.blood_pressure.toFixed(1)}</td>
            <td>${p.age}</td>
            <td style="font-weight: 700; font-family: var(--font-mono); color: ${p.risk_color};">
              ${p.probability}%
            </td>
            <td>
              <span class="pill" style="background: ${p.risk_color}20; color: ${p.risk_color}; border: 1px solid ${p.risk_color}40;">
                ${p.risk_tier}
              </span>
            </td>
            <td style="font-weight: 600; color: var(--neutral-700);">
              ${p.primary_driver}
            </td>
            <td>
              <button class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.6rem;" onclick='loadBatchPatientToForm(${JSON.stringify(p)})'>
                Inspect
              </button>
            </td>
          </tr>
        `;
      }).join('');
    }

  } catch (err) {
    alert('Batch processing failed: ' + err.message);
  }
}

function loadBatchPatientToForm(p) {
  // Switch to diagnostic tab
  const diagTabBtn = document.querySelector('[data-tab="diagnostic-tab"]');
  if (diagTabBtn) diagTabBtn.click();

  document.getElementById('inp-glucose').value = p.glucose;
  document.getElementById('inp-bmi').value = p.bmi;
  document.getElementById('inp-bp').value = p.blood_pressure;
  document.getElementById('inp-age').value = p.age;

  submitPredictionForm();
}
