import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

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
    # 선수별 데이터가 한 행으로 되어 있으므로 그대로 사용 가능
    # style 컬럼 원핫 인코딩 처리
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
    order = np.argsort(-preds)  # 작은 점수가 높은 순위일 수 있으므로 결과 확인 필요
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
    print(f"새 경기 데이터(match_id={next_match_id})가 {CSV_FILE}에 추가되었습니다.")

if __name__ == "__main__":
    print("6명의 선수 능력치와 스타일을 입력하세요.")
    print("예) 3 2 6 1 4 2 훼방꾼")

    players = []
    for i in range(PLAYERS_NUM):
        while True:
            raw = input(f"{i + 1}번 선수: ").strip()
            parts = raw.split()
            if len(parts) != 7:
                print("잘못된 입력입니다. 6개의 숫자와 1개의 스타일명을 입력하세요.")
                continue
            try:
                stats = list(map(int, parts[:6]))
            except:
                print("능력치는 정수여야 합니다.")
                continue
            style = parts[6]
            players.append(stats + [style])
            break

    df = load_training_data()

    if df.empty:
        print("학습 데이터가 없습니다. 예측을 할 수 없습니다.")
        predicted_ranks = list(range(1, PLAYERS_NUM + 1))
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoder.fit(np.array([[p[6]] for p in players]))
    else:
        X, y, groups, encoder = prepare_training_data(df)
        model = train_lgb_ranker(X, y, groups)
        predicted_ranks = predict_ranking(model, players, encoder)

    print("\n[예측 순위 결과]")
    for i, rank in enumerate(predicted_ranks, 1):
        print(f"{i}번 선수 → 예측 순위: {rank}")

    print("\n실제 순위를 입력하세요. (예: 6 1 4 3 2 5)")
    print("입력한 등수는 선수번호 순서가 아닌, 등수 순서대로 선수번호를 의미합니다.")
    while True:
        try:
            input_ranks = list(map(int, input("> ").strip().split()))
            if len(input_ranks) != PLAYERS_NUM:
                print(f"{PLAYERS_NUM}개의 선수번호를 입력하세요.")
                continue
            if sorted(input_ranks) != list(range(1, PLAYERS_NUM + 1)):
                print(f"1부터 {PLAYERS_NUM}까지 선수번호를 모두 포함해야 합니다.")
                continue
            break
        except:
            print("잘못된 입력입니다. 다시 입력하세요.")

    ranks = [0] * PLAYERS_NUM
    for rank_pos, player_num in enumerate(input_ranks, start=1):
        ranks[player_num - 1] = rank_pos

    append_new_match(players, ranks)
