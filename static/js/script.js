// Global Variables
let mapInit = false, moistChart, tempChart, myChart;

// Tab Switcher
function tab(id) {
    document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    
    document.getElementById(id).style.display = 'block';
    
    // Find button and set active
    const buttons = document.getElementsByTagName('button');
    for(let b of buttons) {
        if(b.onclick && b.onclick.toString().includes(id)) b.classList.add('active');
    }

    if(id === 'monitor' && !mapInit) { setTimeout(initMonitor, 200); mapInit = true; }
}

// 1. MONITOR & CHARTS
function initMonitor() {
    const map = L.map('map').setView([20.59, 78.96], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    L.circle([22.5, 77.5], {color:'red',radius:50000}).addTo(map);
    
    moistChart = initChart('chartMoist', '#10b981');
    tempChart = initChart('chartTemp', '#f59e0b');
    
    setInterval(() => {
        const m = Math.floor(Math.random()*20+40), t = Math.floor(Math.random()*10+25);
        updateChart(moistChart, m); updateChart(tempChart, t);
        document.getElementById('val_moist').innerText=m+"%"; 
        document.getElementById('val_temp').innerText=t+"°C";
    }, 2000);
}

function initChart(id, color) {
    const ctx = document.getElementById(id).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(10).fill(''), 
            datasets: [{
                data: Array(10).fill(null), 
                borderColor: color, 
                backgroundColor: color + '20', 
                borderWidth: 2,
                tension: 0.4, 
                pointRadius: 0,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false } },
            scales: { y: { display: false }, x: { display: false } }
        }
    });
}

function updateChart(c,v) { c.data.datasets[0].data.shift(); c.data.datasets[0].data.push(v); c.update(); }

// --- API HELPER ---
function val(id) { return document.getElementById(id).value; }
function show(id, html) { const d=document.getElementById(id); d.style.display='block'; d.innerHTML=html; }

async function apiCall(endpoint, data, resId, format) {
    try {
        const res = await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
        const r = await res.json();
        if(r.error) show(resId, `<span style="color:red">Error: ${r.error}</span>`);
        else show(resId, format(r));
    } catch(e) { console.error(e); }
}

// Bind functions to window so HTML can see them
window.predictYield = () => apiCall('/predict_yield', {Temp:val('y_temp'), Humidity:val('y_hum'), pH:val('y_ph'), Rainfall:val('y_rain'), Crop:val('y_crop'), Model_Type:val('y_model')}, 'y_res', r => `<h2>${r.Result} kg/ha</h2><small>${r.Model}</small>`);
window.recommendCrop = () => apiCall('/recommend_crop', {N:val('r_n'), P:val('r_p'), K:val('r_k'), Temp:val('r_temp'), Humidity:val('r_hum'), pH:val('r_ph'), Rainfall:val('r_rain')}, 'r_res', r => `<h2>${r.Result}</h2>`);
window.checkRisk = () => apiCall('/predict_risk', {Rainfall:val('risk_rain')}, 'risk_res', r => `<h3 style="color:${r.Color}">${r.Status}</h3>`);
window.checkStorage = () => apiCall('/storage_life', {Temp:val('st_temp'), Humidity:val('st_hum'), Crop:val('st_crop')}, 'st_res', r => `Shelf Life: <b>${r.Days} Days</b><br>${r.Advice}`);
window.runSim = () => apiCall('/simulate_impact', {Temp:25, Rainfall:val('s_rain'), Crop:val('s_crop'), Temp_Change:val('s_t_ch'), Rain_Change:val('s_r_ch')}, 's_res', r => `Original: ${r.Original} <br> New: <b>${r.New}</b> <br> Impact: ${r.Percent}%`);
window.getRotation = () => apiCall('/suggest_rotation', {N:val('rot_n')}, 'rot_res', r => `Next: <b>${r.NextCrop}</b>`);
window.smartWater = () => apiCall('/smart_irrigation', {Temp:val('w_temp'), Humidity:val('w_hum'), Rainfall:val('w_rain')}, 'w_res', r => `<b>${r.Status}</b><br>${r.Advice}`);
window.checkPrice = () => apiCall('/market_price', {Crop:val('m_crop')}, 'm_res', r => `Price: <b>₹${r.Price} / Quintal</b>`);
window.calcFinance = () => apiCall('/financial_calc', {Yield:val('fin_yield'), Price:val('fin_price'), Cost:val('fin_cost')}, 'fin_res', r => `Profit: <b>₹${r.Profit}</b>`);
window.getCalendar = () => apiCall('/crop_calendar', {Crop:val('cal_crop'), Date:val('cal_date')}, 'cal_res', r => `Harvest: <b>${r.Harvest}</b>`);
window.getFert = () => apiCall('/fertilizer_advice', {N:val('f_n'), P:val('f_p'), K:val('f_k'), Crop:val('f_crop')}, 'f_res', r => r.Result);

async function analyzeGraph() {
    const res = await fetch('/analyze_sensitivity', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({Crop:val('a_crop'), Parameter:val('a_param')})});
    const r = await res.json();
    const ctx = document.getElementById('yieldChart').getContext('2d');
    if(myChart) myChart.destroy();
    myChart = new Chart(ctx, {type:'line', data:{labels:r.Data.map(d=>d.x), datasets:[{label:r.Label, data:r.Data.map(d=>d.y), borderColor:'#10b981', fill:true, backgroundColor:'rgba(16, 185, 129, 0.1)'}]}});
}

// Bind analyzeGraph to window
window.analyzeGraph = analyzeGraph;