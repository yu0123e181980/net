const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');

let splashWindow;
let mainWindow;

/**
 * スプラッシュウィンドウを作成
 */
function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 600,
    height: 400,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  splashWindow.loadFile(path.join(__dirname, 'frontend', 'splash.html'));
  splashWindow.center();

  // スプラッシュ画面を2.5秒後に閉じてメインウィンドウを表示
  setTimeout(() => {
    createMainWindow();
    if (splashWindow) {
      splashWindow.close();
      splashWindow = null;
    }
  }, 2500);
}

/**
 * メインウィンドウを作成
 */
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    show: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    },
    icon: path.join(__dirname, '../assets/icon.png')
  });

  mainWindow.loadFile(path.join(__dirname, 'frontend', 'index.html'));

  // ウィンドウの準備ができたら表示
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.center();

    // 開発モードの場合はDevToolsを開く
    if (process.argv.includes('--dev')) {
      mainWindow.webContents.openDevTools();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

/**
 * 必要なディレクトリを作成
 */
function ensureDirectories() {
  const directories = [
    path.join(__dirname, '../data/input'),
    path.join(__dirname, '../data/output'),
    path.join(__dirname, '../logs')
  ];

  directories.forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`ディレクトリを作成しました: ${dir}`);
    }
  });
}

/**
 * ログファイルに記録
 */
function writeLog(message, level = 'INFO') {
  const logDir = path.join(__dirname, '../logs');
  const logFile = path.join(logDir, `app-${new Date().toISOString().split('T')[0]}.log`);
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] [${level}] ${message}\n`;

  fs.appendFileSync(logFile, logMessage, 'utf8');
}

// IPCハンドラー: ログ記録
ipcMain.on('write-log', (event, message, level) => {
  writeLog(message, level);
});

// IPCハンドラー: エラー通知
ipcMain.on('show-error', (event, title, message) => {
  const { dialog } = require('electron');
  dialog.showErrorBox(title, message);
  writeLog(`エラー: ${title} - ${message}`, 'ERROR');
});

// アプリケーション起動時の処理
app.whenReady().then(() => {
  try {
    ensureDirectories();
    writeLog('アプリケーションを起動しました');
    createSplashWindow();
  } catch (error) {
    console.error('起動エラー:', error);
    writeLog(`起動エラー: ${error.message}`, 'ERROR');
  }
});

// すべてのウィンドウが閉じられたときの処理
app.on('window-all-closed', () => {
  writeLog('アプリケーションを終了しました');
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// macOSでアイコンクリック時にウィンドウを再作成
app.on('activate', () => {
  if (mainWindow === null) {
    createSplashWindow();
  }
});

// アプリケーション終了前の処理
app.on('before-quit', () => {
  writeLog('アプリケーション終了処理を開始');
});

// 未処理のエラーをキャッチ
process.on('uncaughtException', (error) => {
  console.error('未処理のエラー:', error);
  writeLog(`未処理のエラー: ${error.message}\n${error.stack}`, 'CRITICAL');
});
