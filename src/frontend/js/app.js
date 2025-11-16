/**
 * カテゴリーマネジメントツール - フロントエンドアプリケーション
 *
 * 主な機能:
 * - ビュー切り替えナビゲーション
 * - ファイルアップロード（ドラッグ&ドロップ対応）
 * - プログレスバー表示
 * - トースト通知
 * - Chart.js によるグラフ描画
 */

// ===========================
// グローバル変数
// ===========================
let selectedFiles = [];
let charts = {};

// ===========================
// 初期化
// ===========================
document.addEventListener('DOMContentLoaded', () => {
  console.log('アプリケーションを初期化中...');

  initializeNavigation();
  initializeFileUpload();
  initializeCharts();

  showToast('success', '準備完了', 'カテゴリーマネジメントツールを起動しました');

  console.log('初期化完了');
});

// ===========================
// ナビゲーション
// ===========================
function initializeNavigation() {
  const navItems = document.querySelectorAll('.nav-item');

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const viewName = item.getAttribute('data-view');
      navigateToView(viewName);
    });
  });
}

function navigateToView(viewName) {
  // すべてのナビゲーションアイテムから active クラスを削除
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.remove('active');
  });

  // クリックされたナビゲーションアイテムに active クラスを追加
  const activeNav = document.querySelector(`[data-view="${viewName}"]`);
  if (activeNav) {
    activeNav.classList.add('active');
  }

  // すべてのビューを非表示
  document.querySelectorAll('.view-content').forEach(view => {
    view.classList.remove('active');
  });

  // 選択されたビューを表示
  const targetView = document.getElementById(`view-${viewName}`);
  if (targetView) {
    targetView.classList.add('active');
    console.log(`ビューを切り替え: ${viewName}`);
  }
}

// ===========================
// ファイルアップロード
// ===========================
function initializeFileUpload() {
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const btnImport = document.getElementById('btn-import');
  const btnClearFiles = document.getElementById('btn-clear-files');

  if (!uploadZone || !fileInput) return;

  // クリックでファイル選択
  uploadZone.addEventListener('click', () => {
    fileInput.click();
  });

  // ファイル選択時
  fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
  });

  // ドラッグ&ドロップ
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    handleFiles(files);
  });

  // インポートボタン
  if (btnImport) {
    btnImport.addEventListener('click', () => {
      importFiles();
    });
  }

  // クリアボタン
  if (btnClearFiles) {
    btnClearFiles.addEventListener('click', () => {
      clearFiles();
    });
  }
}

function handleFiles(files) {
  const allowedExtensions = ['.xlsx', '.xls', '.csv'];

  Array.from(files).forEach(file => {
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (allowedExtensions.includes(ext)) {
      // 重複チェック
      const isDuplicate = selectedFiles.some(f => f.name === file.name && f.size === file.size);

      if (!isDuplicate) {
        selectedFiles.push(file);
        console.log(`ファイルを追加: ${file.name} (${formatFileSize(file.size)})`);
      } else {
        showToast('warning', '重複ファイル', `${file.name} は既に追加されています`);
      }
    } else {
      showToast('error', '無効なファイル', `${file.name} は対応していない形式です`);
    }
  });

  updateFileList();
}

function updateFileList() {
  const fileList = document.getElementById('file-list');
  const btnImport = document.getElementById('btn-import');

  if (!fileList) return;

  if (selectedFiles.length === 0) {
    fileList.classList.remove('has-files');
    fileList.innerHTML = '';
    if (btnImport) btnImport.disabled = true;
    return;
  }

  fileList.classList.add('has-files');
  if (btnImport) btnImport.disabled = false;

  fileList.innerHTML = selectedFiles.map((file, index) => `
    <div class="file-item">
      <div class="file-info">
        <span class="file-icon">📄</span>
        <div class="file-details">
          <h4>${escapeHtml(file.name)}</h4>
          <p>${formatFileSize(file.size)} • ${file.type || '不明'}</p>
        </div>
      </div>
      <button class="file-remove" onclick="removeFile(${index})" title="削除">✕</button>
    </div>
  `).join('');
}

function removeFile(index) {
  const file = selectedFiles[index];
  selectedFiles.splice(index, 1);
  console.log(`ファイルを削除: ${file.name}`);
  updateFileList();
  showToast('info', 'ファイル削除', `${file.name} を削除しました`);
}

function clearFiles() {
  if (selectedFiles.length === 0) return;

  const count = selectedFiles.length;
  selectedFiles = [];
  updateFileList();

  const fileInput = document.getElementById('file-input');
  if (fileInput) fileInput.value = '';

  showToast('info', 'クリア', `${count} 件のファイルをクリアしました`);
}

function importFiles() {
  if (selectedFiles.length === 0) {
    showToast('warning', '警告', 'ファイルが選択されていません');
    return;
  }

  console.log(`${selectedFiles.length} 件のファイルをインポート開始`);

  // プログレスバーを表示
  showProgress('データ取込中...', 'ファイルを処理しています');

  // シミュレーション: 実際にはバックエンドに送信
  let progress = 0;
  const interval = setInterval(() => {
    progress += Math.random() * 15;
    if (progress >= 100) {
      progress = 100;
      clearInterval(interval);

      setTimeout(() => {
        hideProgress();
        showToast('success', '完了', `${selectedFiles.length} 件のファイルを取り込みました`);
        clearFiles();
      }, 500);
    }

    updateProgress(progress, `ファイルを処理中... (${Math.floor(progress)}%)`);
  }, 200);
}

// ===========================
// プログレスバー
// ===========================
function showProgress(title = '処理中...', message = '') {
  const modal = document.getElementById('progress-modal');
  const titleElement = document.getElementById('progress-title');
  const messageElement = document.getElementById('progress-message');

  if (modal) {
    modal.classList.add('show');
    if (titleElement) titleElement.textContent = title;
    if (messageElement) messageElement.textContent = message;
    updateProgress(0);
  }
}

function updateProgress(percentage, message = '') {
  const progressBar = document.getElementById('progress-bar');
  const percentageElement = document.getElementById('progress-percentage');
  const messageElement = document.getElementById('progress-message');

  const value = Math.min(Math.max(percentage, 0), 100);

  if (progressBar) {
    progressBar.style.width = `${value}%`;
  }

  if (percentageElement) {
    percentageElement.textContent = `${Math.floor(value)}%`;
  }

  if (message && messageElement) {
    messageElement.textContent = message;
  }
}

function hideProgress() {
  const modal = document.getElementById('progress-modal');
  if (modal) {
    modal.classList.remove('show');
  }
}

// ===========================
// トースト通知
// ===========================
function showToast(type = 'info', title = '', message = '', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const icons = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ'
  };

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || icons.info}</span>
    <div class="toast-content">
      <div class="toast-title">${escapeHtml(title)}</div>
      ${message ? `<div class="toast-message">${escapeHtml(message)}</div>` : ''}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
  `;

  container.appendChild(toast);

  // 自動削除
  if (duration > 0) {
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(400px)';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  console.log(`[${type.toUpperCase()}] ${title}: ${message}`);
}

// ===========================
// グラフ描画
// ===========================
function initializeCharts() {
  // サンプルデータでグラフを初期化
  createSalesChart();
  createProfitChart();
}

function createSalesChart() {
  const canvas = document.getElementById('sales-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');

  charts.salesChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
      datasets: [{
        label: '売上高（百万円）',
        data: [120, 190, 150, 180, 220, 200],
        backgroundColor: 'rgba(102, 126, 234, 0.8)',
        borderColor: 'rgba(102, 126, 234, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        title: {
          display: true,
          text: '月別売上推移',
          font: {
            size: 16
          }
        },
        legend: {
          display: true,
          position: 'bottom'
        }
      },
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });
}

function createProfitChart() {
  const canvas = document.getElementById('profit-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');

  charts.profitChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
      datasets: [
        {
          label: '粗利益（百万円）',
          data: [30, 45, 35, 50, 60, 55],
          borderColor: 'rgba(118, 75, 162, 1)',
          backgroundColor: 'rgba(118, 75, 162, 0.1)',
          tension: 0.4,
          fill: true
        },
        {
          label: '営業利益（百万円）',
          data: [15, 25, 20, 30, 35, 32],
          borderColor: 'rgba(16, 185, 129, 1)',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          tension: 0.4,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        title: {
          display: true,
          text: '月別利益推移',
          font: {
            size: 16
          }
        },
        legend: {
          display: true,
          position: 'bottom'
        }
      },
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });
}

// ===========================
// ユーティリティ関数
// ===========================
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };

  return text.replace(/[&<>"']/g, m => map[m]);
}

// ===========================
// エラーハンドリング
// ===========================
window.addEventListener('error', (event) => {
  console.error('エラーが発生しました:', event.error);
  showToast('error', 'エラー', event.message || 'アプリケーションエラーが発生しました');
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('未処理のPromise拒否:', event.reason);
  showToast('error', 'エラー', '非同期処理でエラーが発生しました');
});

// ===========================
// グローバルに公開する関数
// ===========================
window.navigateToView = navigateToView;
window.removeFile = removeFile;

console.log('app.js を読み込みました');
