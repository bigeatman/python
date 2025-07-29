from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

app = Flask(__name__)

CSV_FILE = "player_training_data.csv"
ATTRIBUTES = ["speed", "stamina", "power", "guts", "intelligence", "condition"]
PLAYERS_NUM = 6

def load_training_data():
    if not os.path.exists(CSV_FILE):
        cols = ["match_id", "player_id"] + ATTRIBUTES + ["style", "rank"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(CSV_FILE)
    return df

def prepare_training_data(df, encoder=None):
    X = df[ATTRIBUTES + ["style"]]
    y = PLAYERS_NUM + 1 - df["rank"].values
    groups = df.groupby("match_id").size().to_list()

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(X[["style"]])
    style_enc = encoder.transform(X[["style"]])

    X_num = X[ATTRIBUTES].values
    X_encoded = np.hstack([X_num, style_enc])

    return X_encoded, y, groups, encoder

def train_lgb_ranker(X, y, groups):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [1, 3, 6],
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 1,
        "verbose": -1,
        "random_state": 42,
    }
    train_data = lgb.Dataset(X, label=y, group=groups)
    model = lgb.train(params, train_data, num_boost_round=100)
    return model

def prepare_input_features(players, encoder):
    stats = []
    styles = []
    for p in players:
        stats.append(p[:6])
        styles.append([p[6]])
    stats = np.array(stats)
    style_enc = encoder.transform(pd.DataFrame(styles, columns=["style"]))
    X_input = np.hstack([stats, style_enc])
    return X_input

def predict_ranking(model, players, encoder):
    X_input = prepare_input_features(players, encoder)
    preds = model.predict(X_input)
    order = np.argsort(-preds)
    ranks = [0] * PLAYERS_NUM
    for rank, idx in enumerate(order, 1):
        ranks[idx] = rank
    return ranks

def append_new_match(players, ranks):
    df_existing = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame()
    next_match_id = df_existing['match_id'].max() + 1 if not df_existing.empty else 1

    new_rows = []
    for i, player in enumerate(players):
        row = {
            'match_id': next_match_id,
            'player_id': i + 1,
            'speed': player[0],
            'stamina': player[1],
            'power': player[2],
            'guts': player[3],
            'intelligence': player[4],
            'condition': player[5],
            'style': player[6],
            'rank': ranks[i]
        }
        new_rows.append(row)

    df_new = pd.DataFrame(new_rows)
    cols_order = ['match_id', 'player_id', 'speed', 'stamina', 'power', 'guts', 'intelligence', 'condition', 'style', 'rank']
    df_new = df_new[cols_order]

    df_new.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))


HTML_FORM = """
<!doctype html>
<title>Player Ranking Predictor</title>
<h2>6명의 선수 능력치와 스타일을 입력하세요.</h2>
<p>예) 3 2 6 1 4 2 훼방꾼</p>

<form method="post" action="/predict">
  {% for i in range(1,7) %}
    <label>{{i}}번 선수:</label><br>
    <input type="text" name="player{{i}}" required><br><br>
  {% endfor %}
  <input type="submit" value="순위 예측">
</form>

{% if predicted_ranks %}
  <h3>[예측 순위 결과]</h3>
  <ul>
  {% for i, rank in enumerate(predicted_ranks, 1) %}
    <li>{{i}}번 선수 → 예측 순위: {{rank}}</li>
  {% endfor %}
  </ul>

  <h3>실제 순위를 입력하세요. (예: 6 1 4 3 2 5)</h3>
  <form method="post" action="/submit_ranks">
    <input type="hidden" name="players_data" value="{{ players_data }}">
    <input type="text" name="actual_ranks" required>
    <input type="submit" value="저장">
  </form>
{% endif %}

{% if message %}
  <p>{{message}}</p>
{% endif %}
"""

from urllib.parse import quote, unquote
import json

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_FORM)

@app.route("/predict", methods=["POST"])
def predict():
    players = []
    for i in range(1, PLAYERS_NUM + 1):
        raw = request.form.get(f"player{i}", "").strip()
        parts = raw.split()
        if len(parts) != 7:
            return render_template_string(HTML_FORM, message=f"{i}번 선수 입력 오류: 6개의 숫자와 1개의 스타일명을 입력하세요.")
        try:
            stats = list(map(int, parts[:6]))
        except:
            return render_template_string(HTML_FORM, message=f"{i}번 선수 능력치는 정수여야 합니다.")
        style = parts[6]
        players.append(stats + [style])

    df = load_training_data()

    if df.empty:
        predicted_ranks = list(range(1, PLAYERS_NUM + 1))
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoder.fit(np.array([[p[6]] for p in players]))
    else:
        X, y, groups, encoder = prepare_training_data(df)
        model = train_lgb_ranker(X, y, groups)
        predicted_ranks = predict_ranking(model, players, encoder)

    # players 정보를 JSON 직렬화 후 URL 안전하게 인코딩해서 hidden input에 넘김
    players_data = quote(json.dumps(players))

    return render_template_string(HTML_FORM, predicted_ranks=predicted_ranks, players_data=players_data)

@app.route("/submit_ranks", methods=["POST"])
def submit_ranks():
    players_data = request.form.get("players_data", "")
    actual_ranks_raw = request.form.get("actual_ranks", "").strip()

    if not players_data or not actual_ranks_raw:
        return render_template_string(HTML_FORM, message="선수 정보나 실제 순위가 누락되었습니다.")

    players_json = json.loads(unquote(players_data))
    input_ranks = actual_ranks_raw.split()

    if len(input_ranks) != PLAYERS_NUM:
        return render_template_string(HTML_FORM, message=f"{PLAYERS_NUM}개의 선수번호를 입력하세요.")

    try:
        input_ranks = list(map(int, input_ranks))
    except:
        return render_template_string(HTML_FORM, message="잘못된 입력입니다. 선수번호는 정수여야 합니다.")

    if sorted(input_ranks) != list(range(1, PLAYERS_NUM + 1)):
        return render_template_string(HTML_FORM, message=f"1부터 {PLAYERS_NUM}까지 선수번호를 모두 포함해야 합니다.")

    ranks = [0] * PLAYERS_NUM
    for rank_pos, player_num in enumerate(input_ranks, start=1):
        ranks[player_num - 1] = rank_pos

    append_new_match(players_json, ranks)

    return render_template_string(HTML_FORM, message="새 경기 데이터가 저장되었습니다. 다시 입력해주세요.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)