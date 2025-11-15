from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import json
import io
import os

app = Flask(__name__)

# データをグローバル変数として読み込み
# スクリプトと同じディレクトリにあるCSVファイルを読み込む
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '共起分析用.csv')

try:
    df = pd.read_csv(csv_path, encoding='shift-jis')
    df['日付'] = pd.to_datetime(df['日付'].astype(str), format='%Y%m%d')
    print(f"✓ データファイルを読み込みました: {csv_path}")
    print(f"✓ データ件数: {len(df)}行")
except FileNotFoundError:
    print(f"エラー: データファイルが見つかりません: {csv_path}")
    print("共起分析用.csv をこのスクリプトと同じディレクトリに配置してください。")
    exit(1)
except Exception as e:
    print(f"エラー: データファイルの読み込みに失敗しました: {e}")
    exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/hierarchies', methods=['GET'])
def get_hierarchies():
    """利用可能な階層とその項目を取得"""
    hierarchies = {
        'ディビジョン': sorted(df['ディビジョン'].unique().tolist()),
        'ライン': sorted(df['ライン'].unique().tolist()),
        '部門': sorted(df['部門'].unique().tolist()),
        'カテゴリー': sorted(df['カテゴリー'].unique().tolist())
    }
    return jsonify(hierarchies)

@app.route('/api/date_range', methods=['GET'])
def get_date_range():
    """データの日付範囲を取得"""
    return jsonify({
        'min_date': df['日付'].min().strftime('%Y-%m-%d'),
        'max_date': df['日付'].max().strftime('%Y-%m-%d')
    })

def calculate_cooccurrence(filtered_df, hierarchy_col, items, mode, start_date=None, end_date=None):
    """共起分析を実行"""
    
    # 日付フィルタリング（期間併売の場合）
    if mode == 'period' and start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        filtered_df = filtered_df[(filtered_df['日付'] >= start) & (filtered_df['日付'] <= end)]
    
    # 選択された項目でフィルタリング
    filtered_df = filtered_df[filtered_df[hierarchy_col].isin(items)]
    
    # 売上金額を商品ごとに集計
    sales_by_item = filtered_df.groupby(hierarchy_col)['売上税抜金額(円)'].sum().to_dict()
    
    # 顧客×商品のマトリックスを作成
    if mode == 'daily':
        # 同時併売: 顧客ID × 日付 × 商品
        customer_item_matrix = filtered_df.groupby(['顧客ID', '日付', hierarchy_col]).size().reset_index()
        customer_item_matrix = customer_item_matrix.groupby(['顧客ID', '日付'])[hierarchy_col].apply(list).reset_index()
    else:
        # 期間併売: 顧客ID × 商品
        customer_item_matrix = filtered_df.groupby(['顧客ID', hierarchy_col]).size().reset_index()
        customer_item_matrix = customer_item_matrix.groupby('顧客ID')[hierarchy_col].apply(list).reset_index()
    
    # 共起ペアをカウント
    cooccurrence = {}
    item_counts = {}
    total_transactions = len(customer_item_matrix)
    
    for idx, row in customer_item_matrix.iterrows():
        item_list = row[hierarchy_col]
        # ユニークな商品のみを対象
        unique_items = list(set(item_list))
        
        # 各商品の出現回数をカウント
        for item in unique_items:
            item_counts[item] = item_counts.get(item, 0) + 1
        
        # 2商品の組み合わせをカウント
        if len(unique_items) >= 2:
            for item1, item2 in combinations(sorted(unique_items), 2):
                pair = (item1, item2)
                cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
    
    # 共起データをリスト化し、リフト値と信頼度を計算
    results = []
    for (item1, item2), count in cooccurrence.items():
        support_a = item_counts.get(item1, 0) / total_transactions
        support_b = item_counts.get(item2, 0) / total_transactions
        support_ab = count / total_transactions
        
        # リフト値 = P(A∩B) / (P(A) * P(B))
        lift = support_ab / (support_a * support_b) if support_a * support_b > 0 else 0
        
        # 信頼度 = P(B|A) = P(A∩B) / P(A)
        confidence_a_to_b = support_ab / support_a if support_a > 0 else 0
        confidence_b_to_a = support_ab / support_b if support_b > 0 else 0
        
        results.append({
            'item1': item1,
            'item2': item2,
            'count': count,
            'lift': round(lift, 3),
            'confidence_a_to_b': round(confidence_a_to_b, 3),
            'confidence_b_to_a': round(confidence_b_to_a, 3),
            'sales1': sales_by_item.get(item1, 0),
            'sales2': sales_by_item.get(item2, 0)
        })
    
    return results, sales_by_item, item_counts

def get_parent_hierarchy(hierarchy_level):
    """指定された階層の1つ上の階層を取得"""
    hierarchy_order = ['ディビジョン', 'ライン', '部門', 'カテゴリー']
    try:
        index = hierarchy_order.index(hierarchy_level)
        if index > 0:
            return hierarchy_order[index - 1]
    except:
        pass
    return None

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """共起分析を実行してネットワーク図用のデータを返す"""
    data = request.json
    
    hierarchy_level = data.get('hierarchy')
    selected_items = data.get('items', [])
    mode = data.get('mode', 'daily')  # 'daily' or 'period'
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    min_cooccurrence = data.get('min_cooccurrence', 1)
    min_lift = data.get('min_lift', 0)
    
    # 階層カラム名を取得
    hierarchy_col = hierarchy_level
    
    # 共起分析を実行
    results, sales_by_item, item_counts = calculate_cooccurrence(
        df.copy(), hierarchy_col, selected_items, mode, start_date, end_date
    )
    
    # フィルタリング
    filtered_results = [
        r for r in results 
        if r['count'] >= min_cooccurrence and r['lift'] >= min_lift
    ]
    
    # 親階層の情報を取得（色分け用）
    parent_hierarchy = get_parent_hierarchy(hierarchy_level)
    parent_mapping = {}
    
    if parent_hierarchy:
        parent_col = parent_hierarchy
        for item in selected_items:
            parent_val = df[df[hierarchy_col] == item][parent_col].iloc[0] if len(df[df[hierarchy_col] == item]) > 0 else 'その他'
            parent_mapping[item] = parent_val
    else:
        # 親階層がない場合は全て同じグループ
        for item in selected_items:
            parent_mapping[item] = hierarchy_level
    
    # ノードデータを作成
    nodes = []
    node_items = set()
    
    for r in filtered_results:
        node_items.add(r['item1'])
        node_items.add(r['item2'])
    
    for item in node_items:
        nodes.append({
            'id': item,
            'label': item,
            'size': sales_by_item.get(item, 0),
            'group': parent_mapping.get(item, 'その他'),
            'count': item_counts.get(item, 0)
        })
    
    # エッジデータを作成
    edges = []
    for r in filtered_results:
        edges.append({
            'source': r['item1'],
            'target': r['item2'],
            'count': r['count'],
            'lift': r['lift'],
            'confidence_a_to_b': r['confidence_a_to_b'],
            'confidence_b_to_a': r['confidence_b_to_a']
        })
    
    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'parent_hierarchy': parent_hierarchy or hierarchy_level
    })

@app.route('/api/export', methods=['POST'])
def export_data():
    """フィルタリングされた共起データをCSVとしてエクスポート"""
    data = request.json
    
    hierarchy_level = data.get('hierarchy')
    selected_items = data.get('items', [])
    mode = data.get('mode', 'daily')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    min_cooccurrence = data.get('min_cooccurrence', 1)
    min_lift = data.get('min_lift', 0)
    
    hierarchy_col = hierarchy_level
    
    # 共起分析を実行
    results, _, _ = calculate_cooccurrence(
        df.copy(), hierarchy_col, selected_items, mode, start_date, end_date
    )
    
    # フィルタリング
    filtered_results = [
        r for r in results 
        if r['count'] >= min_cooccurrence and r['lift'] >= min_lift
    ]
    
    # DataFrameに変換
    export_df = pd.DataFrame(filtered_results)
    if len(export_df) > 0:
        export_df = export_df[['item1', 'item2', 'count', 'lift', 'confidence_a_to_b', 'confidence_b_to_a']]
        export_df.columns = ['商品1', '商品2', '共起回数', 'リフト値', '信頼度(1→2)', '信頼度(2→1)']
    
    # CSVとして出力
    output = io.StringIO()
    export_df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8-sig')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'cooccurrence_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
