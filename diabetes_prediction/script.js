const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const resultCard = document.getElementById('resultCard');
const resultContent = document.getElementById('resultContent');
const btnText = predictBtn.querySelector('.btn-text');
const btnLoader = predictBtn.querySelector('.btn-loader');

const fieldNames = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
];

// Helper to determine Flask server URL vs local hosting
const getApiUrl = () => {
    if (window.location.protocol === 'file:' || window.location.port !== '5000') {
        return 'http://127.0.0.1:5000/predict';
    }
    return '/predict';
};

// Trained SVM model parameters (weights, mean, variance) for client-side evaluation
const MODEL_PARAMS = {
    mean: [3.8450520833333335, 120.89453125, 69.10546875, 20.536458333333332, 79.79947916666667, 31.992578124999998, 0.47187630208333325, 33.240885416666664],
    scale: [3.3673836124089958, 31.95179590820272, 19.343201628981696, 15.941828626496939, 115.16894926467262, 7.87902573154013, 0.3311128160286291, 11.752572645994181],
    coef: [0.3069273072994652, 1.0099456770962323, -0.2114678284335585, 0.0003657197521751421, -0.16754678431091063, 0.616582733334222, 0.2551605138318631, 0.0738895857005939],
    intercept: -0.7382065636069491
};

// Evaluates prediction locally if python server is offline
function predictLocally(formData) {
    const features = [
        formData.Pregnancies,
        formData.Glucose,
        formData.BloodPressure,
        formData.SkinThickness,
        formData.Insulin,
        formData.BMI,
        formData.DiabetesPedigreeFunction,
        formData.Age
    ];

    let val = MODEL_PARAMS.intercept;
    for (let i = 0; i < features.length; i++) {
        const stdFeature = (features[i] - MODEL_PARAMS.mean[i]) / MODEL_PARAMS.scale[i];
        val += stdFeature * MODEL_PARAMS.coef[i];
    }
    return val >= 0 ? 'Diabetic' : 'Non-Diabetic';
}

form.addEventListener('submit', async function (e) {
    e.preventDefault();

    document.querySelectorAll('.input-error').forEach(el => {
        el.classList.remove('input-error');
    });

    let hasError = false;
    const formData = {};

    fieldNames.forEach(name => {
        const input = form.querySelector(`[name="${name}"]`);
        const value = input.value.trim();

        if (value === '') {
            input.classList.add('input-error');
            hasError = true;
        } else {
            formData[name] = parseFloat(value);
        }
    });

    if (hasError) {
        form.style.animation = 'none';
        void form.offsetWidth;
        form.style.animation = 'shake 0.4s ease';
        return;
    }

    predictBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    resultCard.style.display = 'none';

    try {
        let prediction;
        let isLocalResult = false;

        try {
            const response = await fetch(getApiUrl(), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error(`Status ${response.status}`);
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }
            prediction = data.prediction;
        } catch (serverErr) {
            console.warn("Falling back to local client prediction:", serverErr);
            prediction = predictLocally(formData);
            isLocalResult = true;
        }

        showResult(prediction, isLocalResult);

    } catch (error) {
        resultCard.style.display = 'block';
        resultCard.className = 'card result-card';
        resultContent.innerHTML = `
            <div class="result-icon">❌</div>
            <div class="result-text" style="color:#dc2626;">Error occurred</div>
            <div class="result-sub">${error.message}</div>
        `;
    } finally {
        predictBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

function showResult(prediction, isLocal = false) {
    resultCard.style.display = 'block';
    resultCard.style.animation = 'none';
    void resultCard.offsetWidth;
    resultCard.style.animation = 'fadeSlideUp 0.4s ease-out';

    const infoBadge = isLocal 
        ? `<div class="info-badge" style="margin-top: 12px; font-size: 0.75rem; color: #4b5563; background-color: #f3f4f6; padding: 4px 8px; border-radius: 6px; display: inline-flex; align-items: center; gap: 4px; font-weight: 500; border: 1px dashed #d1d5db;">
            ⚡ Local ML Engine (No Server Required)
           </div>`
        : '';

    if (prediction === 'Diabetic') {
        resultCard.className = 'card result-card result-diabetic';
        resultContent.innerHTML = `
            <div class="result-icon">⚠️</div>
            <div class="result-text">Diabetic</div>
            <div class="result-sub">The model predicts the patient is likely diabetic.</div>
            ${infoBadge}
        `;
    } else {
        resultCard.className = 'card result-card result-nondiabetic';
        resultContent.innerHTML = `
            <div class="result-icon">✅</div>
            <div class="result-text">Non-Diabetic</div>
            <div class="result-sub">The model predicts the patient is likely non-diabetic.</div>
            ${infoBadge}
        `;
    }
}

resetBtn.addEventListener('click', function () {
    form.reset();
    resultCard.style.display = 'none';
    document.querySelectorAll('.input-error').forEach(el => {
        el.classList.remove('input-error');
    });
});
