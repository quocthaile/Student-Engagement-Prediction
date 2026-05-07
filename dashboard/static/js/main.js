// Dashboard Logic
let benchmarkChart, distributionChart, importanceChart, radarChart, classMetricsChart, distributionChartAnalysis;
let predictionHistory = JSON.parse(localStorage.getItem('predict_history') || '[]');

// --- State Management ---
let totalItems = 0;
let currentPage = 1;
const itemsPerPage = 10;
let searchTerm = '';
let currentTestData = []; // Store current page data globally

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initTabs();
    refreshData();
    initPredictForm();
    renderHistory();
    initModelSelector();
    initSearch();
});

// --- Model Selection ---
function initModelSelector() {
    const selector = document.getElementById('model-selector');
    if (!selector) return;

    selector.addEventListener('change', async () => {
        const modelType = selector.value;
        const overlay = document.getElementById('loading-overlay');
        
        overlay.style.display = 'flex';
        
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: modelType })
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                await refreshData();
                showToast(result.message, 'success');
            } else {
                showToast('Lỗi: ' + result.message, 'error');
            }
        } catch (err) {
            console.error('Training error:', err);
            showToast('Có lỗi xảy ra trong quá trình huấn luyện.', 'error');
        } finally {
            overlay.style.display = 'none';
        }
    });
}

// --- Theme Logic ---
function initTheme() {
    const themeToggle = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            setTheme(newTheme);
        });
    }
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    const icon = document.querySelector('#theme-toggle i');
    if (icon) {
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
    
    if (benchmarkChart) updateChartTheme();
}

function updateChartTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#f9fafb' : '#1e293b';
    const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)';

    const charts = [benchmarkChart, distributionChart, importanceChart, radarChart, classMetricsChart, distributionChartAnalysis];
    charts.forEach(chart => {
        if (!chart) return;
        if (chart.options.scales) {
            Object.values(chart.options.scales).forEach(scale => {
                if (scale.grid) scale.grid.color = gridColor;
                if (scale.ticks) scale.ticks.color = isDark ? '#9ca3af' : '#64748b';
                if (scale.angleLines) scale.angleLines.color = gridColor;
            });
        }
        if (chart.options.plugins && chart.options.plugins.legend) {
            chart.options.plugins.legend.labels.color = textColor;
        }
        chart.update();
    });
}

// --- Navigation ---
function initTabs() {
    const navLinks = document.querySelectorAll('.nav-link');
    const tabContents = document.querySelectorAll('.tab-content');

    const tabNames = {
        'overview': 'Tổng quan Dashboard',
        'analysis': 'Phân tích Chi tiết',
        'test-data': 'Kiểm thử Mô hình',
        'predict': 'Dự báo Trực tiếp'
    };

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = link.getAttribute('data-tab');

            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            tabContents.forEach(content => content.style.display = 'none');
            const targetTab = document.getElementById(`tab-${tabId}`);
            if (targetTab) targetTab.style.display = 'block';
            
            document.getElementById('page-title').innerText = tabNames[tabId] || 'Dashboard';
            
            if (tabId === 'test-data') fetchTestData();
        });
    });
}

// --- Data Fetching ---
async function refreshData() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        updateStats(data);
        updateCharts(data);
        
        const modelNameEl = document.getElementById('sidebar-model-name');
        const selector = document.getElementById('model-selector');
        if (data.metadata) {
            if (modelNameEl) modelNameEl.innerText = data.metadata.model_name || 'Linear SVC';
            if (selector && data.metadata.model_type) selector.value = data.metadata.model_type;
            
            // Update train/test sizes
            const stats = data.metadata.data_stats || {};
            if (document.getElementById('stat-train-size')) 
                document.getElementById('stat-train-size').innerText = stats.train_size?.toLocaleString() || '--';
            if (document.getElementById('stat-test-size')) 
                document.getElementById('stat-test-size').innerText = stats.test_size?.toLocaleString() || '--';
        }
    } catch (err) {
        console.error('Error fetching dashboard data:', err);
    }
}

async function fetchTestData() {
    try {
        const response = await fetch(`/api/test-data?page=${currentPage}&page_size=${itemsPerPage}&search=${encodeURIComponent(searchTerm)}`);
        const result = await response.json();
        
        totalItems = result.total;
        currentTestData = result.data; // Store it
        renderTable(currentTestData);
        updatePaginationControls();
    } catch (err) {
        console.error('Error fetching test data:', err);
    }
}

function updateStats(data) {
    const meta = data.metadata || {};
    const metrics = meta.test_metrics || {};
    const elements = {
        'stat-accuracy': metrics.Accuracy,
        'stat-recall': metrics.Recall_Low_Engagement,
        'stat-precision': metrics.Precision_Low_Engagement,
        'stat-f1': metrics.F1_Weighted
    };
    
    for (const [id, val] of Object.entries(elements)) {
        const el = document.getElementById(id);
        if (el) el.innerText = val ? `${(val * 100).toFixed(1)}%` : '--%';
    }
}

function updateCharts(data) {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#f9fafb' : '#1e293b';
    const meta = data.metadata || {};
    const metrics = meta.test_metrics || {};
    const distributions = meta.distributions || { true: {}, pred: {} };

    // 1. Radar Chart (Performance)
    const ctxRadar = document.getElementById('radarChart')?.getContext('2d');
    if (ctxRadar) {
        if (radarChart) radarChart.destroy();
        radarChart = new Chart(ctxRadar, {
            type: 'radar',
            data: {
                labels: ['Độ chính xác', 'Recall (Thấp)', 'Recall (TB)', 'Recall (Cao)', 'F1-Score'],
                datasets: [{
                    label: 'Chỉ số Mô hình',
                    data: [
                        metrics.Accuracy || 0,
                        metrics.Recall_Low_Engagement || 0,
                        metrics.Recall_Medium || 0,
                        metrics.Recall_High || 0,
                        metrics.F1_Weighted || 0
                    ],
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderColor: '#6366f1',
                    pointBackgroundColor: '#6366f1',
                    borderWidth: 2
                }]
            },
            options: {
                scales: { r: { angleLines: { color: 'rgba(255,255,255,0.1)' }, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { display: false } } },
                plugins: { legend: { labels: { color: textColor } } }
            }
        });
    }

    // 2. Importance Chart
    const ctxImp = document.getElementById('importanceChart')?.getContext('2d');
    if (ctxImp) {
        const impData = meta.feature_importance || [];
        const featureMap = {
            'school_encoded': 'Trường học',
            'seq': 'Tổng thời gian học',
            'speed': 'Tốc độ xem video TB',
            'rep_counts': 'Số lần phản hồi',
            'cmt_counts': 'Số lần bình luận',
            'age': 'Tuổi',
            'gender_encoded': 'Giới tính',
            'num_courses': 'Số lượng khóa học',
            'attempts_4w': 'Số lần nộp bài',
            'is_correct_4w': 'Kết quả làm bài',
            'score_4w': 'Điểm số',
            'accuracy_rate_4w': 'Tỷ lệ chính xác'
        };
        
        if (importanceChart) importanceChart.destroy();
        importanceChart = new Chart(ctxImp, {
            type: 'bar',
            data: {
                labels: impData.map(i => featureMap[i.feature] || i.feature),
                datasets: [{
                    label: 'Mức độ ảnh hưởng',
                    data: impData.map(i => i.importance),
                    backgroundColor: '#10b981',
                    borderRadius: 8
                }]
            },
            options: {
                indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: { 
                    x: { grid: { display: false }, ticks: { color: isDark ? '#9ca3af' : '#64748b' } }, 
                    y: { grid: { display: false }, ticks: { color: isDark ? '#9ca3af' : '#64748b' } } 
                }
            }
        });
    }

    // 3. Class Metrics Chart
    const ctxClass = document.getElementById('classMetricsChart')?.getContext('2d');
    if (ctxClass) {
        const cmData = meta.class_metrics || {};
        if (classMetricsChart) classMetricsChart.destroy();
        classMetricsChart = new Chart(ctxClass, {
            type: 'bar',
            data: {
                labels: ['Thấp', 'Trung bình', 'Cao'],
                datasets: [
                    { label: 'Precision', data: [cmData.Low?.precision, cmData.Medium?.precision, cmData.High?.precision], backgroundColor: '#6366f1' },
                    { label: 'Recall', data: [cmData.Low?.recall, cmData.Medium?.recall, cmData.High?.recall], backgroundColor: '#10b981' },
                    { label: 'F1-Score', data: [cmData.Low?.['f1-score'], cmData.Medium?.['f1-score'], cmData.High?.['f1-score']], backgroundColor: '#f59e0b' }
                ]
            },
            options: { scales: { y: { beginAtZero: true, max: 1 } }, plugins: { legend: { labels: { color: textColor } } } }
        });
    }

    // 4. Confusion Matrix
    const cmGrid = document.getElementById('confusion-matrix-grid');
    if (cmGrid) {
        const cmMatrix = meta.confusion_matrix || [[0,0,0],[0,0,0],[0,0,0]];
        cmGrid.innerHTML = '';
        cmMatrix.flat().forEach(val => {
            const cell = document.createElement('div');
            cell.style.padding = '1rem';
            cell.style.background = 'var(--surface-light)';
            cell.style.borderRadius = '8px';
            cell.style.fontWeight = '700';
            cell.style.color = val > 0 ? 'var(--primary-light)' : 'var(--text-muted)';
            cell.innerText = val.toLocaleString();
            cmGrid.appendChild(cell);
        });
    }

    // 5. Real Data Distribution (True vs Pred)
    const ctxDistAn = document.getElementById('distributionChartAnalysis')?.getContext('2d');
    if (ctxDistAn) {
        const labels = ['Low_Engagement', 'Medium_Engagement', 'High_Engagement'];
        if (distributionChartAnalysis) distributionChartAnalysis.destroy();
        distributionChartAnalysis = new Chart(ctxDistAn, {
            type: 'bar',
            data: {
                labels: ['Thấp', 'Trung bình', 'Cao'],
                datasets: [
                    {
                        label: 'Thực tế',
                        data: labels.map(l => distributions.true[l] || 0),
                        backgroundColor: 'rgba(99, 102, 241, 0.5)',
                        borderColor: '#6366f1',
                        borderWidth: 1
                    },
                    {
                        label: 'Dự đoán',
                        data: labels.map(l => distributions.pred[l] || 0),
                        backgroundColor: 'rgba(16, 185, 129, 0.5)',
                        borderColor: '#10b981',
                        borderWidth: 1
                    }
                ]
            },
            options: { 
                scales: { y: { beginAtZero: true, max: 1, ticks: { callback: v => (v*100) + '%' } } },
                plugins: { legend: { labels: { color: textColor } } } 
            }
        });
    }

    // 6. Doughnut Distribution
    const ctxDist = document.getElementById('distributionChart')?.getContext('2d');
    if (ctxDist) {
        const distData = distributions.pred;
        if (distributionChart) distributionChart.destroy();
        distributionChart = new Chart(ctxDist, {
            type: 'doughnut',
            data: {
                labels: ['Thấp', 'Trung bình', 'Cao'],
                datasets: [{
                    data: [distData.Low_Engagement || 0, distData.Medium_Engagement || 0, distData.High_Engagement || 0],
                    backgroundColor: ['#f43f5e', '#f59e0b', '#10b981'],
                    borderWidth: 0
                }]
            },
            options: { plugins: { legend: { position: 'bottom', labels: { color: textColor } } }, cutout: '70%' }
        });
    }

    // 7. Benchmark Chart
    const ctxBench = document.getElementById('benchmarkChart')?.getContext('2d');
    if (ctxBench) {
        const benchData = data.all_models || [];
        if (benchmarkChart) benchmarkChart.destroy();
        benchmarkChart = new Chart(ctxBench, {
            type: 'bar',
            data: {
                labels: benchData.map(b => b.name),
                datasets: [
                    { label: 'Accuracy', data: benchData.map(b => b.accuracy), backgroundColor: '#6366f1', borderRadius: 8 },
                    { label: 'F1-Score', data: benchData.map(b => b.f1), backgroundColor: '#f59e0b', borderRadius: 8 }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { labels: { color: textColor } } },
                scales: { y: { beginAtZero: true, max: 1 } }
            }
        });
    }
}

function renderTable(pageData) {
    const tbody = document.getElementById('test-table-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (!pageData || pageData.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 3rem; color: var(--text-muted);">Không tìm thấy dữ liệu phù hợp</td></tr>`;
    } else {
        pageData.forEach((row, index) => {
            const tr = document.createElement('tr');
            const isMatch = row.y_true_label === row.y_pred_label;
            tr.innerHTML = `
                <td>#${row.row_id}</td>
                <td><span class="badge" style="background: rgba(255,255,255,0.05);">${translateLabel(row.y_true_label)}</span></td>
                <td><span class="badge badge-${getBadgeClass(row.y_pred_label)}">${translateLabel(row.y_pred_label)}</span></td>
                <td>
                    <i class="fas ${isMatch ? 'fa-check-circle' : 'fa-exclamation-triangle'}" 
                       style="color: ${isMatch ? 'var(--secondary)' : 'var(--danger)'}"></i>
                    <span style="margin-left: 0.5rem;">${isMatch ? 'Chính xác' : 'Sai lệch'}</span>
                </td>
                <td><button class="btn" onclick="viewDetails(${index})" style="padding: 0.5rem; background: var(--surface-light); border: 1px solid var(--border);"><i class="fas fa-eye"></i></button></td>
            `;
            tbody.appendChild(tr);
        });
    }
}

function updatePaginationControls() {
    const info = document.getElementById('pagination-info');
    const controls = document.getElementById('pagination-controls');
    if (!info || !controls) return;
    
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    const start = totalItems === 0 ? 0 : (currentPage - 1) * itemsPerPage + 1;
    const end = Math.min(currentPage * itemsPerPage, totalItems);
    
    info.innerText = `${start}-${end} trong ${totalItems}`;
    
    controls.innerHTML = '';
    
    const prevBtn = document.createElement('button');
    prevBtn.className = 'btn';
    prevBtn.disabled = currentPage === 1;
    prevBtn.innerHTML = '<i class="fas fa-chevron-left"></i>';
    prevBtn.onclick = () => { currentPage--; fetchTestData(); };
    controls.appendChild(prevBtn);
    
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);
    if (endPage - startPage < 4) startPage = Math.max(1, endPage - 4);
    
    for (let i = startPage; i <= endPage; i++) {
        const btn = document.createElement('button');
        btn.className = `btn ${i === currentPage ? 'btn-primary' : ''}`;
        btn.innerText = i;
        btn.onclick = () => { currentPage = i; fetchTestData(); };
        controls.appendChild(btn);
    }
    
    const nextBtn = document.createElement('button');
    nextBtn.className = 'btn';
    nextBtn.disabled = currentPage === totalPages || totalPages === 0;
    nextBtn.innerHTML = '<i class="fas fa-chevron-right"></i>';
    nextBtn.onclick = () => { currentPage++; fetchTestData(); };
    controls.appendChild(nextBtn);
}

function initSearch() {
    const searchInput = document.getElementById('search-test-id');
    if (!searchInput) return;
    
    let timeout;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            searchTerm = e.target.value.trim();
            currentPage = 1;
            fetchTestData();
        }, 500);
    });
}

function viewDetails(index) {
    const item = currentTestData[index];
    if (!item) return;

    const modal = document.getElementById('detail-modal');
    const title = document.getElementById('modal-title');
    const body = document.getElementById('modal-body');
    const badge = document.getElementById('modal-prediction-badge');
    
    if (!modal || !body) return;
    
    title.innerHTML = `<i class="fas fa-user-graduate" style="color: var(--primary);"></i> Chi tiết Sinh viên #${item.row_id}`;
    
    const featureNames = {
        'school': 'Trường học (Gốc)',
        'gender': 'Giới tính (Gốc)',
        'year_of_birth': 'Năm sinh',
        'age': 'Tuổi',
        'seq': 'Tổng thời gian học (seq)',
        'speed': 'Tốc độ xem video TB (speed)',
        'rep_counts': 'Số lần phản hồi (rep)',
        'cmt_counts': 'Số lần bình luận',
        'num_courses': 'Số lượng khóa học',
        'attempts_4w': 'Số lần nộp bài (4 tuần)',
        'is_correct_4w': 'Kết quả làm bài (4 tuần)',
        'score_4w': 'Điểm số (4 tuần)',
        'accuracy_rate_4w': 'Tỷ lệ chính xác'
    };

    body.innerHTML = '';
    
    // Priority order for display
    const displayOrder = ['school', 'gender', 'year_of_birth', 'age', 'num_courses', 'seq', 'speed', 'rep_counts', 'cmt_counts', 'attempts_4w', 'is_correct_4w', 'score_4w', 'accuracy_rate_4w'];
    
    displayOrder.forEach(key => {
        if (item[key] === undefined && key !== 'school' && key !== 'gender') return;
        
        const label = featureNames[key] || key;
        let val = item[key];
        
        // Use original values if available
        if (key === 'school' && !val) val = item['school_name'];
        if (key === 'gender' && (val === undefined || val === null)) {
             const gMap = {0: 'Nam', 1: 'Nữ', 2: 'Khác'};
             val = gMap[item['gender_encoded']] || 'N/A';
        }

        const formattedVal = typeof val === 'number' ? (val % 1 === 0 ? val : val.toFixed(2)) : (val || '--');
        
        const div = document.createElement('div');
        div.innerHTML = `
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.25rem;">${label}</p>
            <p style="font-weight: 600; font-size: 1rem;">${formattedVal}</p>
        `;
        body.appendChild(div);
    });
    
    const resultsDiv = document.createElement('div');
    resultsDiv.style.gridColumn = 'span 2';
    resultsDiv.style.marginTop = '1rem';
    resultsDiv.style.padding = '1rem';
    resultsDiv.style.background = 'rgba(255,255,255,0.02)';
    resultsDiv.style.borderRadius = '12px';
    resultsDiv.style.display = 'flex';
    resultsDiv.style.gap = '2rem';
    resultsDiv.innerHTML = `
        <div>
            <p style="font-size: 0.8rem; color: var(--text-muted);">Kết quả Thực tế</p>
            <p style="font-weight: 700; color: var(--text);">${translateLabel(item.y_true_label)}</p>
        </div>
        <div>
            <p style="font-size: 0.8rem; color: var(--text-muted);">Kết quả Dự đoán</p>
            <p style="font-weight: 700; color: var(--primary-light);">${translateLabel(item.y_pred_label)}</p>
        </div>
    `;
    body.appendChild(resultsDiv);
    
    badge.innerHTML = `<span class="badge badge-${getBadgeClass(item.y_pred_label)}">${translateLabel(item.y_pred_label)}</span>`;
    modal.style.display = 'flex';
}

// --- Custom Notifications ---
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icons = {
        'success': 'fa-check-circle',
        'error': 'fa-exclamation-circle',
        'info': 'fa-info-circle',
        'warning': 'fa-exclamation-triangle'
    };

    toast.innerHTML = `
        <i class="fas ${icons[type] || 'fa-info-circle'}" style="font-size: 1.25rem;"></i>
        <div style="flex: 1;">${message}</div>
    `;

    container.appendChild(toast);

    // Auto-remove after 5s
    const timer = setTimeout(() => {
        toast.classList.add('toast-fade-out');
        setTimeout(() => toast.remove(), 400);
    }, 5000);

    toast.onclick = () => {
        clearTimeout(timer);
        toast.remove();
    };
}

function closeModal() {
    const modal = document.getElementById('detail-modal');
    if (modal) modal.style.display = 'none';
}

function translateLabel(label) {
    const trans = {'Low_Engagement': 'Thấp', 'Medium_Engagement': 'Trung bình', 'High_Engagement': 'Cao'};
    return trans[label] || label;
}

function getBadgeClass(label) {
    if (label?.includes('Low')) return 'low';
    if (label?.includes('Medium')) return 'medium';
    return 'high';
}

function initPredictForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const data = {};
        formData.forEach((val, key) => {
            if (key === 'school_name') {
                data[key] = val;
            } else if (['speed', 'score_4w', 'accuracy_rate_4w'].includes(key)) {
                data[key] = parseFloat(val);
            } else {
                data[key] = parseInt(val);
            }
        });
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                showPrediction(result.prediction);
                saveToHistory(result.prediction);
            }
        } catch (err) { console.error('Prediction error:', err); }
    });
}

function showPrediction(label) {
    const resultDiv = document.getElementById('prediction-result');
    const valueEl = document.getElementById('prediction-value');
    const badgeEl = document.getElementById('prediction-badge');
    
    if (valueEl) valueEl.innerText = translateLabel(label);
    if (badgeEl) {
        badgeEl.className = `badge badge-${getBadgeClass(label)}`;
        badgeEl.innerText = label === 'Low_Engagement' ? 'Cảnh báo rủi ro' : 'Tương tác tốt';
    }
    
    if (resultDiv) resultDiv.style.display = 'block';
}

function saveToHistory(label) {
    predictionHistory.unshift({ label, time: new Date().toLocaleTimeString() });
    if (predictionHistory.length > 5) predictionHistory.pop();
    localStorage.setItem('predict_history', JSON.stringify(predictionHistory));
    renderHistory();
}

function renderHistory() {
    const container = document.getElementById('predict-history');
    if (!container) return;
    if (predictionHistory.length === 0) return;
    
    container.innerHTML = predictionHistory.map(item => `
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; border-bottom: 1px solid var(--border);">
            <span>${translateLabel(item.label)}</span>
            <span style="opacity: 0.5;">${item.time}</span>
        </div>
    `).join('');
}
