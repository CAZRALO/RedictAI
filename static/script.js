// --- TAB NAVIGATION LOGIC (Giữ nguyên) ---
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        this.classList.add('active');
        document.getElementById(this.dataset.tab).classList.add('active');
    });
});

// --- AI DIAGNOSIS LOGIC (Đã cập nhật) ---
const uploadForm = document.getElementById('uploadForm');
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');
const diagnoseBtnDeep = document.getElementById('diagnoseBtnDeep');
const diagnoseBtnFast = document.getElementById('diagnoseBtnFast');
const loader = document.getElementById('loader');
const resultDiv = document.getElementById('prediction-result');
const uploadBox = document.getElementById('uploadBox');
const uploadAnotherBtn = document.getElementById('uploadAnotherBtn');
const resultCard = document.getElementById('resultCard');
const chartContainer = document.getElementById('chart-container');
const chartCanvas = document.getElementById('resultChart');
const scoreCamBtn = document.getElementById('scoreCamBtn');
const scoreCamLoader = document.getElementById('scoreCamLoader');
const scoreCamError = document.getElementById('scoreCamError');

let resultChartInstance = null;

// *** Biến toàn cục để lưu trữ ảnh và kết quả cho ScoreCAM ***
let storedOriginalImageB64 = null;
let storedPredictionLabel = null;

function handleFileSelect(files) {
    if (files && files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreviewContainer.style.display = 'flex';
            diagnoseBtnDeep.disabled = false;
            diagnoseBtnFast.disabled = false;
            uploadBox.style.display = 'none';
        }
        reader.readAsDataURL(files[0]);
    }
}

imageUpload.addEventListener('change', function() {
    handleFileSelect(this.files);
});

uploadBox.addEventListener('dragover', (e) => { e.preventDefault(); uploadBox.style.borderColor = 'var(--accent-laser-cyan)'; });
uploadBox.addEventListener('dragleave', (e) => { e.preventDefault(); uploadBox.style.borderColor = 'var(--accent-hyper-violet)'; });
uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = 'var(--accent-hyper-violet)';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageUpload.files = files;
        handleFileSelect(files);
    }
});

diagnoseBtnDeep.addEventListener('click', async function(e) {
    e.preventDefault();
    await runPrediction('deep');
});

diagnoseBtnFast.addEventListener('click', async function(e) {
    e.preventDefault();
    await runPrediction('fast');
});

async function runPrediction(mode) {
    const formData = new FormData();
    formData.append('image', imageUpload.files[0]);
    formData.append('mode', mode);
    
    resultCard.style.display = 'block';
    
    // --- START MODIFICATION ---
    // Hiển thị loader và xóa kết quả cũ
    loader.style.display = 'block';
    resultDiv.innerHTML = ''; // Xóa kết quả cũ
    resultDiv.style.display = 'none'; // Ẩn div kết quả

    // Chỉ hiển thị thông báo chờ nếu là chế độ 'deep'
    if (mode === 'deep') {
        resultDiv.innerHTML = '<p style="color: var(--text-muted); margin-top: 1rem;">Đang thực hiện chẩn đoán sâu...<br>Quá trình này có thể mất 1-2 phút, vui lòng không rời khỏi trang.</p>';
        resultDiv.style.display = 'block'; // Hiển thị thông báo
    }
    // --- END MODIFICATION ---
    
    chartContainer.style.display = 'none';
    document.getElementById('image-results-container').style.display = 'none'; // Ẩn ảnh khi bắt đầu
    scoreCamBtn.style.display = 'none'; // Ẩn nút ScoreCAM
    scoreCamLoader.style.display = 'none';
    scoreCamError.style.display = 'none';
    diagnoseBtnDeep.disabled = true;
    diagnoseBtnFast.disabled = true;

    // Xóa dữ liệu cũ
    storedOriginalImageB64 = null;
    storedPredictionLabel = null;
    
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.statusText}`);
        }
        const data = await response.json();
        displayResult(data, mode); // Hàm này sẽ tự động ghi đè thông báo chờ
    } catch (error) {
        console.error('Diagnosis error:', error);
        displayResult({ error: error.message }, mode); // Hàm này cũng sẽ ghi đè thông báo chờ
    } finally {
        loader.style.display = 'none'; // Ẩn loader sau khi hoàn tất
        diagnoseBtnDeep.disabled = false;
        diagnoseBtnFast.disabled = false;
    }
}

uploadAnotherBtn.addEventListener('click', function() {
    imageUpload.value = '';
    imagePreview.src = '#';
    imagePreviewContainer.style.display = 'none';
    uploadBox.style.display = 'flex';
    diagnoseBtnDeep.disabled = true;
    diagnoseBtnFast.disabled = true;
    resultCard.style.display = 'none';
    resultDiv.innerHTML = '<p>Kết quả sẽ được hiển thị tại đây.</p>';
    resultDiv.style.display = 'block';
    chartContainer.style.display = 'none';
    
    // THÊM: Reset và ẩn container ảnh kết quả
    document.getElementById('image-results-container').style.display = 'none';
    document.getElementById('originalResultImage').src = '';
    document.getElementById('heatmapResultImage').src = '';
    
    // Ẩn và reset nút ScoreCAM
    scoreCamBtn.style.display = 'none';
    scoreCamLoader.style.display = 'none';
    scoreCamError.style.display = 'none';
    storedOriginalImageB64 = null;
    storedPredictionLabel = null;
    
    resultCard.className = 'card result-area';
    if(resultChartInstance) {
        resultChartInstance.destroy();
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

function displayResult(data, mode) {
    let resultHTML = '';
    resultCard.className = 'card result-area';

    // THÊM: Lấy elements của container ảnh
    const imageResultsContainer = document.getElementById('image-results-container');
    const originalResultImage = document.getElementById('originalResultImage');
    const heatmapResultImage = document.getElementById('heatmapResultImage');
    const heatmapColumn = document.getElementById('heatmap-column');

    if (data.error) {
        resultHTML = `<div style="color: var(--vote-red);">Lỗi: ${data.error}</div>`;
        resultCard.classList.add('vote-red');
        imageResultsContainer.style.display = 'none'; // Ẩn ảnh nếu có lỗi
        scoreCamBtn.style.display = 'none'; // Ẩn nút ScoreCAM
    } else if (data.prediction) {
        const { prediction, vote_percentage, color, chart_data, vote_distribution } = data;

        resultCard.classList.add(`vote-${color}`);
        
        let mainResultHTML;
        if (mode === 'fast') {
            mainResultHTML = `
                <div style="font-size: 1.5rem; color: var(--text-starlight);">Kết quả nhanh: <span style="color: var(--vote-${color}); font-weight: 700;">${prediction}</span></div>
                <div style="font-size: 1rem; color: var(--text-starlight); margin-top: 0.5rem;">
                    Độ tự tin: ${vote_percentage.toFixed(2)}%
                </div>
            `;
        } else {
             mainResultHTML = `
                <div style="font-size: 1.5rem; color: var(--text-starlight);">Kết quả: <span style="color: var(--vote-${color}); font-weight: 700;">${prediction}</span></div>
                <div style="font-size: 1rem; color: var(--text-starlight); margin-top: 0.5rem;">
                    Tỷ lệ đồng thuận: ${vote_percentage.toFixed(2)}%
                </div>
            `;
        }
        
        let distributionHTML = '';
        if (mode === 'deep' && vote_distribution && vote_distribution.length > 0) {
            distributionHTML = '<div style="margin-top: 1.5rem; font-size: 0.9rem; text-align: left; width: 100%; max-width: 300px; border-top: 1px solid var(--border-color); padding-top: 1rem;">';
            distributionHTML += '<h4 style="color: var(--text-starlight); margin-bottom: 0.5rem; text-align: center;">Chi tiết biểu quyết:</h4>';
            vote_distribution.forEach(item => {
                distributionHTML += `<p style="color: var(--text-starlight);">${item.label}: ${item.percentage.toFixed(2)}%</p>`;
            });
            distributionHTML += '</div>';
        }

        resultHTML = mainResultHTML + distributionHTML;
        
        // --- CẬP NHẬT LOGIC HIỂN THỊ ẢNH ---
        if (data.original_image_b64) { 
            originalResultImage.src = 'data:image/jpeg;base64,' + data.original_image_b64;
            imageResultsContainer.style.display = 'block'; // Hiển thị container
            
            // Lưu trữ dữ liệu để dùng cho ScoreCAM
            storedOriginalImageB64 = data.original_image_b64;
            storedPredictionLabel = data.prediction;

            // Kiểm tra heatmap (có thể null nếu Grad-CAM lỗi)
            if (data.heatmap_image_b64) {
                heatmapResultImage.src = 'data:image/jpeg;base64,' + data.heatmap_image_b64;
                heatmapColumn.style.display = 'block'; // Hiển thị div heatmap
                scoreCamBtn.style.display = 'block'; // Hiển thị nút ScoreCAM
            } else {
                heatmapResultImage.src = '';
                heatmapColumn.style.display = 'none'; // Ẩn div heatmap
                scoreCamBtn.style.display = 'none'; // Ẩn nút ScoreCAM nếu heatmap mặc định lỗi
            }
        } else {
            imageResultsContainer.style.display = 'none'; // Ẩn container nếu không có ảnh gốc
            scoreCamBtn.style.display = 'none';
        }
        
        if (mode === 'deep') {
            chartContainer.style.display = 'block';
            drawScatterPlot(chart_data);
        } else {
            chartContainer.style.display = 'none';
            if(resultChartInstance) {
                resultChartInstance.destroy();
            }
        }
    }

    resultDiv.innerHTML = resultHTML;
    resultDiv.style.display = 'block';
}

// *** THÊM EVENT LISTENER CHO NÚT SCORECAM ***
scoreCamBtn.addEventListener('click', async () => {
    if (!storedOriginalImageB64 || !storedPredictionLabel) {
        scoreCamError.textContent = 'Lỗi: Không tìm thấy dữ liệu ảnh gốc.';
        scoreCamError.style.display = 'block';
        return;
    }
    
    // Hiển thị loader, ẩn nút, reset lỗi
    scoreCamLoader.style.display = 'block';
    scoreCamBtn.style.display = 'none';
    scoreCamError.style.display = 'none';
    
    const heatmapResultImage = document.getElementById('heatmapResultImage');
    heatmapResultImage.classList.add('loading'); // Làm mờ ảnh heatmap cũ

    try {
        const response = await fetch('/generate-heatmap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_b64: storedOriginalImageB64,
                prediction_label: storedPredictionLabel,
                heatmap_method: 'ScoreCAM'
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.heatmap_image_b64) {
            // Cập nhật ảnh heatmap
            heatmapResultImage.src = 'data:image/jpeg;base64,' + data.heatmap_image_b64;
            // Cập nhật tiêu đề
            document.getElementById('heatmap-column').querySelector('h4').textContent = 'Heatmap (ScoreCAM)';
        } else {
            throw new Error('Không nhận được heatmap từ server.');
        }
        
    } catch (error) {
        console.error('ScoreCAM error:', error);
        scoreCamError.textContent = `Lỗi tạo heatmap chi tiết: ${error.message}`;
        scoreCamError.style.display = 'block';
    } finally {
        // Hoàn tất: Ẩn loader, hiện lại nút
        scoreCamLoader.style.display = 'none';
        scoreCamBtn.style.display = 'block';
        heatmapResultImage.classList.remove('loading');
    }
});


function drawScatterPlot(data) {
    if (resultChartInstance) {
        resultChartInstance.destroy();
    }

    const validData = Array.isArray(data) ? data.filter(p => p != null && p.x !== undefined && p.y !== undefined) : [];
    const trueVotesData = validData.filter(p => p.vote_true);
    const falseVotesData = validData.filter(p => !p.vote_true);
    
    // --- START MODIFICATION ---
    // Đổi từ biến CSS 'var(--text-starlight)' sang giá trị HEX
    // để đảm bảo Chart.js có thể đọc được màu.
    const chartTextColor = '#ccd6f6'; // Giá trị của --text-starlight
    // --- END MODIFICATION ---

    const chartGridColor = 'rgba(204, 214, 246, 0.1)';

    const ctx = chartCanvas.getContext('2d');
    resultChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                // *** SỬA LỖI: Thêm kiểm tra context.raw trong hàm radius ***
                { label: 'Vote True (Đồng thuận)', data: trueVotesData, backgroundColor: 'var(--vote-green)', borderColor: '#10B981', borderWidth: 2, radius: (context) => (context.raw && context.raw.x === 1) ? 7 : 5, hoverRadius: 9, },
                { label: 'Vote False (Không đồng thuận)', data: falseVotesData, backgroundColor: 'var(--vote-yellow)', borderColor: '#F59E0B', borderWidth: 2, radius: (context) => (context.raw && context.raw.x === 1) ? 7 : 5, hoverRadius: 9, }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: chartTextColor, font: { size: 14 } } },
                tooltip: {
                    callbacks: {
                        // *** SỬA LỖI: Thêm kiểm tra context.raw trong hàm label ***
                        label: function(context) {
                            // Thêm kiểm tra context.raw
                            if (!context || !context.raw) {
                                return 'Dữ liệu không xác định';
                            }
                            let voteType = context.dataset.label.split(' ')[0];
                            let runLabel = context.raw.x === 1 ? 'Ảnh gốc' : `Lần lọc ${context.raw.x - 1}`;
                            // Thêm kiểm tra context.raw.y
                            let yValue = (typeof context.raw.y === 'number') ? context.raw.y.toFixed(2) : 'N/A';
                            return `${voteType} - ${runLabel}: ${yValue}%`;
                        }
                    }
                },
                zoom: { pan: { enabled: false }, zoom: { wheel: { enabled: false }, drag: { enabled: false }, pinch: { enabled: false }, mode: 'xy' } }
            },
            scales: {
                x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Lần dự đoán (1: Ảnh gốc, 2-51: Ảnh đã lọc)', color: chartTextColor }, ticks: { color: chartTextColor, stepSize: 5 }, min: 1, max: 51, grid: { color: chartGridColor } },
                y: { title: { display: true, text: 'Độ chính xác Pred (%)', color: chartTextColor }, ticks: { color: chartTextColor }, min: 0, max: 100, grid: { color: chartGridColor } }
            }
        }
    });
}

// --- CHATBOT & FACT-CHECKER LOGIC (CẬP NHẬT) ---
const chatBody = document.getElementById('chatBody');
const chatInput = document.getElementById('chat-input');
// const chatFileInput = document.getElementById('chatFileInput'); // ĐÃ XÓA
// let chatAttachedImageBase64 = null; // ĐÃ XÓA

function handleChatKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

/* // ĐÃ XÓA EventListener cho chatFileInput
chatFileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            chatAttachedImageBase64 = e.target.result.split(',')[1];
            appendChatMessage(`Đã đính kèm ảnh: <b>${file.name}</b>`, 'user');
        };
        reader.readAsDataURL(file);
    }
});
*/

async function sendChatMessage() {
    const messageText = chatInput.value.trim();
    // if (!messageText && !chatAttachedImageBase64) return; // ĐÃ SỬA
    if (!messageText) return; // Chỉ kiểm tra text

    if (messageText) {
        appendChatMessage(messageText, 'user');
    }
    chatInput.value = '';
    chatInput.focus();

    appendChatMessage("...", 'bot', true);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: messageText
                // image_data: chatAttachedImageBase64 // ĐÃ XÓA
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Lỗi từ server: ${response.statusText}`);
        }

        const data = await response.json();
        updateLastBotMessage(data.response);

    } catch (error) {
        console.error('Lỗi khi gửi tin nhắn:', error);
        updateLastBotMessage(`Xin lỗi, đã có lỗi xảy ra: ${error.message}`);
    } finally {
        // chatAttachedImageBase64 = null; // ĐÃ XÓA
        // chatFileInput.value = ''; // ĐÃ XÓA
    }
}

async function sendFactCheckRequest() {
    const factText = chatInput.value.trim();

    if (!factText) {
        chatInput.classList.add('input-error');
        chatInput.placeholder = "Vui lòng nhập thông tin cần kiểm chứng!";
        setTimeout(() => {
            chatInput.classList.remove('input-error');
            chatInput.placeholder = "Nhập câu hỏi hoặc thông tin cần kiểm chứng...";
        }, 1500);
        return;
    }

    appendChatMessage(`<b>[Yêu cầu kiểm chứng]</b><br>${factText.replace(/</g, "&lt;").replace(/>/g, "&gt;")}`, 'user');
    chatInput.value = '';
    chatInput.focus();

    appendChatMessage("Đang kiểm chứng thông tin...", 'bot', true);

    try {
        const response = await fetch('/fact-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fact: factText })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Lỗi từ server: ${response.statusText}`);
        }

        const data = await response.json();
        if (data.success) {
            const formattedResult = formatFactCheckResult(data.result);
            updateLastBotMessage(formattedResult);
        } else {
            throw new Error(data.error || "Phản hồi không thành công.");
        }

    } catch (error) {
        console.error('Lỗi khi kiểm chứng thông tin:', error);
        updateLastBotMessage(`Xin lỗi, đã có lỗi xảy ra khi kiểm chứng: ${error.message}`);
    }
}

// --- CÁC HÀM MỚI VÀ HÀM ĐƯỢC CẬP NHẬT ---

async function factCheckThisMessage(buttonElement) {
    const messageElement = buttonElement.parentElement;
    const messageWrapper = messageElement.parentElement;

    if (messageWrapper.querySelector('.fact-check-inline-container')) {
        messageWrapper.querySelector('.fact-check-inline-container').remove();
        return;
    }

    const messageClone = messageElement.cloneNode(true);
    messageClone.querySelector('.fact-check-btn').remove();
    
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = messageClone.innerHTML;
    const textToVerify = tempDiv.textContent || tempDiv.innerText || "";
    
    if (!textToVerify.trim()) {
        console.error("Không có nội dung để kiểm chứng.");
        return;
    }
    
    buttonElement.disabled = true;

    const resultContainer = document.createElement('div');
    resultContainer.className = 'fact-check-inline-container';
    resultContainer.innerHTML = `<div class="chat-message bot-message" style="opacity:0.8;">Đang kiểm chứng...</div>`;
    messageWrapper.appendChild(resultContainer);
    chatBody.scrollTop = chatBody.scrollHeight;

    try {
        const response = await fetch('/fact-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fact: textToVerify })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Lỗi từ server: ${response.statusText}`);
        }

        const data = await response.json();
        if (data.success) {
            resultContainer.innerHTML = formatFactCheckResult(data.result);
        } else {
            throw new Error(data.error || "Phản hồi không thành công.");
        }

    } catch (error) {
        console.error('Lỗi khi kiểm chứng thông tin:', error);
        const errorHtml = `
            <div class="fact-check-result">
                <h4 style="color: var(--vote-red);">Lỗi</h4>
                <p>Không thể hoàn tất kiểm chứng: ${error.message}</p>
            </div>`;
        resultContainer.innerHTML = errorHtml;
    } finally {
        buttonElement.disabled = false;
        chatBody.scrollTop = chatBody.scrollHeight;
    }
}

function formatFactCheckResult(result) {
    const analysis = result.claim_analysis || {};
    const conclusion = result.conclusion || {};
    const evidence = result.evidence || [];

    let evidenceHTML = '';
    if (evidence.length > 0) {
        evidenceHTML = '<h4>Bằng chứng:</h4>';
        evidence.forEach((item, index) => {
            evidenceHTML += `
                <div class="evidence-item">
                    <p><b>Nguồn ${index + 1}:</b> <a href="${item.source_url || '#'}" target="_blank" rel="noopener noreferrer">${item.source_title || 'Không có tiêu đề'}</a></p>
                    <p><i>"${item.relevant_excerpt || 'Không có trích dẫn'}"</i></p>
                    <p><b>Độ tin cậy:</b> ${item.source_credibility || 'N/A'} | <b>Kết luận nguồn:</b> ${item.supports_claim || 'N/A'}</p>
                </div>
            `;
        });
    }

    return `
        <div class="fact-check-result">
            <h4>Kết quả kiểm chứng</h4>
            <p><b>Tuyên bố:</b> ${analysis.original_claim || 'N/A'}</p>
            <p><b>Trạng thái:</b> <span class="status-${(analysis.verification_status || 'UNVERIFIED').replace(/\s+/g, '_')}">${analysis.verification_status || 'UNVERIFIED'}</span></p>
            <p><b>Tóm tắt:</b> ${analysis.summary || 'N/A'}</p>
            <hr style="border-color: var(--border-color); margin: 10px 0; border-style: solid; border-width: 1px 0 0 0;">
            <p><b>Kết luận cuối cùng:</b> ${conclusion.explanation || 'N/A'}</p>
            <p><b>Khuyến nghị:</b> ${conclusion.recommendation || 'N/A'}</p>
            ${evidenceHTML}
        </div>
    `;
}

function appendChatMessage(text, sender, isTyping = false) {
    const messageWrapper = document.createElement('div');
    // Add wrapper classes for alignment
    messageWrapper.classList.add('message-wrapper', sender);
    
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', `${sender}-message`);

    if (sender === 'user') {
        messageElement.innerHTML = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    } else { // sender === 'bot'
        if (isTyping) {
            messageWrapper.id = 'typing-indicator';
            messageElement.innerHTML = `<div style="display: flex; gap: 4px; align-items: center; justify-content: center; height: 100%;">
                <span style="animation: typing-dot 1.2s infinite ease-in-out both; animation-delay: 0s; width: 6px; height: 6px; background: currentColor; border-radius: 50%;"></span>
                <span style="animation: typing-dot 1.2s infinite ease-in-out both; animation-delay: 0.2s; width: 6px; height: 6px; background: currentColor; border-radius: 50%;"></span>
                <span style="animation: typing-dot 1.2s infinite ease-in-out both; animation-delay: 0.4s; width: 6px; height: 6px; background: currentColor; border-radius: 50%;"></span>
            </div>
            <style> @keyframes typing-dot { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1.0); } } </style>`;
        }
    }
    
    messageWrapper.appendChild(messageElement);
    chatBody.appendChild(messageWrapper);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function updateLastBotMessage(text) {
    const typingWrapper = document.getElementById('typing-indicator');
    if (typingWrapper) {
        typingWrapper.id = '';
        const messageElement = typingWrapper.querySelector('.chat-message');
        if (messageElement) {
            messageElement.innerHTML = text;
            const factCheckBtn = document.createElement('button');
            factCheckBtn.className = 'fact-check-btn';
            factCheckBtn.title = 'Kiểm chứng thông tin này';
            factCheckBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" style="width:16px; height:16px;">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>`;
            factCheckBtn.onclick = function() { factCheckThisMessage(this); };
            messageElement.appendChild(factCheckBtn);
        }
    }
}