import requests
import numpy as np
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamelog, teamestimatedmetrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pickle
import config

API_KEY = config.api_key

def fetch_odds_data(api_key):
    
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    
    #add odds and teams for upcoming games to a dataframe
    params = {
        'api_key': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'bookmakers': ['draftkings']
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception(f"Failed to fetch odds data. Error: {response.text}")
    
    odds_list = []
    for game in data:
        game_data = {
            'game_id': game['id'],
            'team1_name': game['bookmakers'][0]['markets'][0]['outcomes'][0]['name'],
            'team2_name': game['bookmakers'][0]['markets'][0]['outcomes'][1]['name'],
            'team1_odds': game['bookmakers'][0]['markets'][0]['outcomes'][0]['price'],
            'team2_odds': game['bookmakers'][0]['markets'][0]['outcomes'][1]['price']
        }
        odds_list.append(game_data)
    return pd.DataFrame(odds_list)

def fetch_team_stats_data(start_date, end_date):
    #get game logs for all teams in date range
    nba_teams = teams.get_teams()
    game_log = leaguegamelog.LeagueGameLog(date_from_nullable=start_date, date_to_nullable=end_date, league_id='00')
    games = game_log.get_data_frames()[0]
    nba_games = games[games['TEAM_ID'].isin([team['id'] for team in nba_teams])]
    game_log_e = teamestimatedmetrics.TeamEstimatedMetrics(season='2022-23', league_id='00')
    games_e = game_log_e.get_data_frames()[0]
    nba_games_e = games_e[games_e['TEAM_ID'].isin([team['id'] for team in nba_teams])]

    #calculate releveant stats
    team_stats = []
    for team in nba_teams:
        team_id = team['id']
        team_games = nba_games[(nba_games['TEAM_ID'] == team_id)]
        team_games_e = nba_games_e[(nba_games_e['TEAM_ID'] == team_id)]
        wins = len(team_games[team_games['WL'] == 'W'])
        losses = len(team_games[team_games['WL'] == 'L'])
        win_percentage = wins / (wins + losses) if (wins + losses) > 0 else 0
        points = team_games['PTS'].sum()
        games_played = len(team_games)
        average_points = points / games_played if games_played > 0 else 0
        #fg3_pct = team_games['FG3_PCT'].sum() / games_played if games_played > 0 else 0
        #fg2_pct = team_games['FG_PCT'].sum() / games_played if games_played > 0 else 0
        avg_reb = team_games['REB'].sum() / games_played if games_played > 0 else 0
        avg_oreb = team_games['OREB'].sum() / games_played if games_played > 0 else 0
        avg_ast = team_games['AST'].sum() / games_played if games_played > 0 else 0
        avg_tov =team_games['TOV'].sum() / games_played if games_played > 0 else 0
        fgm = team_games['FGM'].sum() / games_played if games_played > 0 else 0
        fg3m = team_games['FG3M'].sum() / games_played if games_played > 0 else 0
        fga = team_games['FGA'].sum() / games_played if games_played > 0 else 0
        efg_pct = (fgm + 0.5*fg3m) / fga
        plus_minus = team_games['PLUS_MINUS'].sum() / games_played if games_played > 0 else 0
        pace = team_games_e["E_PACE"].sum()
        off_rating = team_games_e["E_OFF_RATING"].sum()
        net_rating = team_games_e["E_NET_RATING"].sum()
        ast_ratio = team_games_e["E_AST_RATIO"].sum()
        tov_pct = team_games_e["E_TM_TOV_PCT"].sum()

        #add relevant stats to dict and add a row for each team in a dataframe
        team_stat = {
            'TEAM_ID': team_id,
            'team_name': team['full_name'],
            'games_played': games_played,
            'wins': wins,
            'losses': losses,
            'win_percentage': win_percentage,
            'average_points': average_points,
            'reb':avg_reb,
            'oreb':avg_oreb,
            'ast':avg_ast,
            'tov':avg_tov,
            'plus_minus': plus_minus,
            'efg_pct': efg_pct,
            'pace': pace,
            'off_rating': off_rating,
            'net_rating': net_rating,
            'ast_ratio': ast_ratio,
            'tov_pct':tov_pct
        }
        team_stats.append(team_stat)
    return pd.DataFrame(team_stats)

def preprocess_data(odds_data, team_stats_data):
    #merge odds and stats data for each team playing in upcoming games
    for team in odds_data['team1_name']:
        merged_df = odds_data.merge(team_stats_data, left_on="team1_name", right_on="team_name", suffixes=('', '_team1'))
        merged_df = merged_df.merge(team_stats_data, left_on="team2_name", right_on="team_name", suffixes=('_team1', '_team2'))
        merged_df = merged_df.drop(columns=['TEAM_ID_team1', 'team_name_team1', 'TEAM_ID_team2', 'team_name_team2'])
    return merged_df

def select_features(data):
    #drop non-numerical columns for input matrix
    features_to_drop = ['game_id', 'team1_name', 'team2_name','team1_odds','team2_odds']
    return data.drop(columns=features_to_drop)

def train_stacked_model(X_train, y_train):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    rf_params = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    rf_pipe = Pipeline([('scaler', StandardScaler()), ('classifier', rf)])

    grid_search = GridSearchCV(rf_pipe, rf_params, cv=k_fold, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    lr = LogisticRegression(max_iter=3000, random_state=42)
    lr_pipe = Pipeline([('scaler', StandardScaler()), ('classifier', lr)])

    xgb = XGBClassifier(random_state=42)
    xgb_pipe = Pipeline([('scaler', StandardScaler()), ('classifier', xgb)])

    model = VotingClassifier(
        estimators=[('rf', best_rf), ('lr', lr_pipe), ('xgb', xgb_pipe)],
        voting='soft'
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    #start and end date for training data
    #start_date = "2022-10-19" #start of nba season 2022-23
    start_date = "2023-02-10" #trade deadline 2023
    end_date = "2023-04-09" #current day

    #fetch all data and preproccess
    odds_data = fetch_odds_data(API_KEY)
    team_stats_data = fetch_team_stats_data(start_date, end_date)
    team_ids = team_stats_data['TEAM_ID'].unique()
    data = preprocess_data(odds_data, team_stats_data)

    #set matrix and vector
    X = select_features(data)
    y = np.where(data['team1_odds'] < data['team2_odds'], 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_stacked_model(X_train, y_train)
    predictions = model.predict(X_test)

    # save trained model to .pkl
    model_filename = 'trained_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    print("Model saved to trained_model.pkl")
    print("Use run.py to use it")