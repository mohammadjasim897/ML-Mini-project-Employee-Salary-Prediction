/**
 * SalaryIQ — Frontend Logic
 * Employee Salary Prediction System
 * Handles form interactions, API calls, animated results, and dynamic dropdowns.
 */

// ── DOM Elements ────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const form = $('#predictionForm');
const predictBtn = $('#predictBtn');
const formSection = $('#formSection');
const resultsSection = $('#resultsSection');
const tryAgainBtn = $('#tryAgainBtn');

// Sliders
const sliders = {
    age:            { el: $('#age'),            display: $('#ageValue'),            format: v => `${v} yrs` },
    experience:     { el: $('#experience'),     display: $('#experienceValue'),     format: v => `${parseFloat(v).toFixed(1)} yrs` },
    skills:         { el: $('#skills'),         display: $('#skillsValue'),         format: v => v },
    certifications: { el: $('#certifications'), display: $('#certificationsValue'), format: v => v },
    performance:    { el: $('#performance'),    display: $('#performanceValue'),    format: v => `${parseFloat(v).toFixed(1)} ⭐` },
};

// Dropdowns
const genderSelect = $('#gender');
const educationSelect = $('#education');
const domainSelect = $('#domain');
const jobTitleSelect = $('#jobTitle');
const companySizeSelect = $('#companySize');
const cityTierSelect = $('#cityTier');

// ── State ────────────────────────────────────────────────────────────
let appMetadata = null;

// ── Initialize ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    initSliders();
    await loadMetadata();
});

function initSliders() {
    Object.values(sliders).forEach(({ el, display, format }) => {
        display.textContent = format(el.value);
        el.addEventListener('input', () => {
            display.textContent = format(el.value);
        });
    });
}

async function loadMetadata() {
    try {
        const res = await fetch('/api/metadata');
        appMetadata = await res.json();

        // Populate dropdowns
        populateSelect(genderSelect, appMetadata.genders);
        populateSelect(educationSelect, appMetadata.education_levels);
        populateSelect(domainSelect, appMetadata.domains);
        populateSelect(companySizeSelect, appMetadata.company_sizes);
        populateSelect(cityTierSelect, appMetadata.city_tiers);

        // Model badge
        $('#modelName').textContent = appMetadata.model_name;
        $('#accuracyStat').textContent = `${(appMetadata.r2_score * 100).toFixed(1)}%`;

        // Domain → Job title chaining
        domainSelect.addEventListener('change', () => {
            const domain = domainSelect.value;
            if (domain && appMetadata.domain_jobs[domain]) {
                populateSelect(jobTitleSelect, appMetadata.domain_jobs[domain]);
                jobTitleSelect.disabled = false;
            } else {
                jobTitleSelect.innerHTML = '<option value="">Select domain first</option>';
                jobTitleSelect.disabled = true;
            }
        });

    } catch (err) {
        console.error('Failed to load metadata:', err);
        $('#modelName').textContent = 'Connection error';
    }
}

function populateSelect(selectEl, options) {
    const placeholder = selectEl.options[0]?.textContent || 'Select...';
    selectEl.innerHTML = `<option value="">${placeholder}</option>`;
    options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        selectEl.appendChild(option);
    });
}

// ── Form Submission ──────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const name = $('#name').value.trim();
    if (!name) {
        shakeElement($('#name'));
        return;
    }

    // Validate all selects
    const selects = [genderSelect, educationSelect, domainSelect, jobTitleSelect, companySizeSelect, cityTierSelect];
    for (const sel of selects) {
        if (!sel.value) {
            shakeElement(sel);
            sel.focus();
            return;
        }
    }

    // Show loading
    setLoading(true);

    const payload = {
        name: name,
        age: parseInt(sliders.age.el.value),
        gender: genderSelect.value,
        education_level: educationSelect.value,
        domain: domainSelect.value,
        job_title: jobTitleSelect.value,
        experience_years: parseFloat(sliders.experience.el.value),
        skills_count: parseInt(sliders.skills.el.value),
        certifications: parseInt(sliders.certifications.el.value),
        company_size: companySizeSelect.value,
        city_tier: cityTierSelect.value,
        performance_rating: parseFloat(sliders.performance.el.value),
    };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await res.json();

        if (!res.ok) {
            alert(data.error || 'Prediction failed');
            setLoading(false);
            return;
        }

        displayResults(data, payload);

    } catch (err) {
        console.error('Prediction error:', err);
        alert('Failed to connect to server. Make sure the Flask app is running.');
    } finally {
        setLoading(false);
    }
});

// ── Display Results ──────────────────────────────────────────────────
function displayResults(data, input) {
    // Hide form, show results
    formSection.style.display = 'none';
    $('#hero').style.display = 'none';
    resultsSection.style.display = 'block';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Animate salary counter
    animateCounter($('#salaryAmount'), data.predicted_salary);
    $('#salaryLakhs').textContent = data.salary_lakhs;
    $('#monthlySalary').textContent = data.monthly_salary;
    $('#resultModel').textContent = data.model_used;
    $('#resultAccuracy').textContent = data.model_accuracy;

    // Employee summary
    const genderEmoji = input.gender === 'Male' ? '👨‍💼' : input.gender === 'Female' ? '👩‍💼' : '🧑‍💼';
    $('#summaryAvatar').textContent = genderEmoji;
    $('#summaryName').textContent = input.name;
    $('#summaryRole').textContent = `${input.job_title} • ${input.domain}`;
    $('#summaryText').textContent = data.summary.text;

    // Highlights
    const highlightsContainer = $('#summaryHighlights');
    highlightsContainer.innerHTML = '';
    (data.summary.highlights || []).forEach(h => {
        const tag = document.createElement('span');
        tag.className = 'highlight-tag';
        tag.textContent = h;
        highlightsContainer.appendChild(tag);
    });

    // Feature importance chart
    renderImportanceChart(appMetadata?.feature_importance || []);
}

// ── Animated Counter ─────────────────────────────────────────────────
function animateCounter(el, targetValue) {
    const duration = 1500;
    const startTime = performance.now();
    const startValue = 0;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(startValue + (targetValue - startValue) * eased);

        el.textContent = formatIndianCurrency(current);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function formatIndianCurrency(num) {
    const s = Math.round(num).toString();
    if (s.length <= 3) return '₹' + s;

    let result = s.slice(-3);
    let remaining = s.slice(0, -3);
    while (remaining.length > 0) {
        result = remaining.slice(-2) + ',' + result;
        remaining = remaining.slice(0, -2);
    }
    return '₹' + result;
}

// ── Feature Importance Chart ─────────────────────────────────────────
function renderImportanceChart(features) {
    const container = $('#importanceChart');
    container.innerHTML = '';

    if (!features || features.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">Feature importance not available for this model type.</p>';
        return;
    }

    // Normalize to max
    const maxImp = Math.max(...features.map(f => f.importance));

    features.slice(0, 10).forEach((feat, i) => {
        const pct = (feat.importance / maxImp * 100).toFixed(1);
        const label = cleanFeatureName(feat.feature);

        const bar = document.createElement('div');
        bar.className = 'importance-bar';
        bar.innerHTML = `
            <span class="importance-bar__label" title="${feat.feature}">${label}</span>
            <div class="importance-bar__track">
                <div class="importance-bar__fill" style="width: 0%;"></div>
            </div>
            <span class="importance-bar__value">${(feat.importance * 100).toFixed(1)}%</span>
        `;
        container.appendChild(bar);

        // Animate with delay
        setTimeout(() => {
            bar.querySelector('.importance-bar__fill').style.width = `${pct}%`;
        }, 100 + i * 80);
    });
}

function cleanFeatureName(name) {
    // Clean sklearn feature names: "cat__domain_IT" → "Domain: IT"
    return name
        .replace(/^num__/, '')
        .replace(/^cat__/, '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

// ── Try Again ────────────────────────────────────────────────────────
tryAgainBtn.addEventListener('click', () => {
    resultsSection.style.display = 'none';
    formSection.style.display = 'block';
    $('#hero').style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ── Utilities ────────────────────────────────────────────────────────
function setLoading(loading) {
    const textEl = predictBtn.querySelector('.form__submit-text');
    const loadingEl = predictBtn.querySelector('.form__submit-loading');
    if (loading) {
        textEl.style.display = 'none';
        loadingEl.style.display = 'flex';
        predictBtn.disabled = true;
    } else {
        textEl.style.display = 'flex';
        loadingEl.style.display = 'none';
        predictBtn.disabled = false;
    }
}

function shakeElement(el) {
    el.style.animation = 'none';
    el.offsetHeight; // Trigger reflow
    el.style.animation = 'shake 0.4s ease-in-out';
    el.style.borderColor = '#ef4444';
    setTimeout(() => {
        el.style.borderColor = '';
        el.style.animation = '';
    }, 800);
}

// Add shake keyframes dynamically
const shakeStyle = document.createElement('style');
shakeStyle.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-6px); }
        40% { transform: translateX(6px); }
        60% { transform: translateX(-4px); }
        80% { transform: translateX(4px); }
    }
`;
document.head.appendChild(shakeStyle);
