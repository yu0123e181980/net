# 📊 カテゴリーマネジメントツール

食品小売業界向けのカテゴリーマネジメント支援ツール（Electronデスクトップアプリケーション）

## 🎯 概要

このツールは、食品小売業界におけるカテゴリーマネジメント業務を効率化するために開発されました。POSデータの分析、カテゴリー別の売上・利益分析、棚割計画の最適化などの機能を提供します。

## ✨ 主な機能

- **📥 データ取込**: Excel/CSVファイルからのデータインポート
- **📊 カテゴリー分析**: 売上・利益・在庫状況の可視化
- **🏪 棚割計画**: 最適な棚割プランの作成支援
- **📄 レポート作成**: 分析結果のレポート出力
- **📤 データ出力**: 分析結果のエクスポート

## 🛠️ 技術スタック

### フロントエンド
- **Electron**: デスクトップアプリケーション化
- **HTML5/CSS3/JavaScript (ES6+)**: UI実装
- **Chart.js**: グラフ描画

### バックエンド（今後実装予定）
- **Python 3.9+**: データ処理エンジン
- **pandas**: データ分析
- **scikit-learn**: 機械学習
- **openpyxl**: Excel処理

## 📋 必要要件

- **Node.js**: v18.0.0 以上
- **npm**: v9.0.0 以上

## 🚀 セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd category-management-tool
```

### 2. 依存パッケージのインストール

```bash
npm install
```

### 3. アプリケーションの起動

```bash
npm start
```

開発モードで起動（DevToolsが自動的に開きます）:

```bash
npm run dev
```

## 📦 ビルド

### 全プラットフォーム向けビルド

```bash
npm run build
```

### プラットフォーム別ビルド

```bash
# Windows向け
npm run build:win

# macOS向け
npm run build:mac

# Linux向け
npm run build:linux
```

ビルド成果物は `dist/` ディレクトリに出力されます。

## 📁 プロジェクト構造

```
category-management-tool/
├── src/
│   ├── main.js                 # Electronメインプロセス
│   ├── frontend/
│   │   ├── index.html         # メインウィンドウ
│   │   ├── splash.html        # スプラッシュ画面
│   │   ├── css/
│   │   │   └── style.css      # スタイルシート
│   │   └── js/
│   │       └── app.js         # フロントエンドロジック
│   └── backend/               # バックエンド処理（今後実装）
├── data/
│   ├── input/                 # 入力データ
│   └── output/                # 出力データ
├── logs/                      # アプリケーションログ
├── package.json
└── README.md
```

## 🎨 機能詳細

### スプラッシュ画面
- 起動時に2.5秒間表示
- グラデーション背景とローディングアニメーション
- プログレスバー表示

### ダッシュボード
- 各機能へのクイックアクセス
- カテゴリーマネジメントの概要説明

### データ取込
- ドラッグ&ドロップ対応
- Excel (.xlsx, .xls) / CSV対応
- プログレスバー付き処理状況表示
- エラーハンドリング

### カテゴリー分析
- Chart.jsを使用したグラフ表示
- 売上推移の可視化
- 利益推移の可視化

## 🔧 開発原則

1. **段階的実装**: 機能を独立モジュールとして実装
2. **完全な日本語対応**: Shift_JIS/UTF-8に対応
3. **エラーハンドリング徹底**: すべての処理でエラー処理を実装
4. **ユーザビリティ重視**: 直感的なUI/UX設計
5. **コード可読性**: メンテナンス性の高いコード

## 📝 ログ出力

アプリケーションログは `logs/` ディレクトリに日付別で保存されます。

- ログファイル名: `app-YYYY-MM-DD.log`
- ログレベル: INFO, ERROR, CRITICAL

## 🐛 トラブルシューティング

### アプリケーションが起動しない場合

1. Node.jsのバージョンを確認
   ```bash
   node --version
   ```

2. 依存パッケージを再インストール
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

3. ログファイルを確認
   ```bash
   cat logs/app-YYYY-MM-DD.log
   ```

### ビルドが失敗する場合

- `electron-builder` の再インストール
  ```bash
  npm install electron-builder --save-dev
  ```

## 🔮 今後の実装予定

- [ ] Pythonバックエンドとの連携
- [ ] 高度な統計分析機能
- [ ] 機械学習による需要予測
- [ ] カスタムレポートテンプレート
- [ ] データベース連携
- [ ] クラウド同期機能

## 📄 ライセンス

MIT License

## 👥 開発チーム

Category Management Team

---

**Version**: 1.0.0
**Last Updated**: 2025-11-16
