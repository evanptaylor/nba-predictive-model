import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, teamdashboardbylastngames, playerdashboardbylastngames, leaguegamelog

# Replace with your API key
API_KEY = "72f975515f75626b533a11b8354015e6"

def fetch_odds_data(start_date, end_date, api_key):
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"

    # Add parameters to a dictionary
    params = {
        'api_key': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'start': start_date,
        'end': end_date,
        'bookmakers': ['draftkings']
        #'bookmakers': 1
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


def fetch_team_stats_data():
    nba_teams = teams.get_teams()

    # Fetch NBA game logs for all time
    game_log = leaguegamelog.LeagueGameLog(season='2021-22', league_id='00')
    games = game_log.get_data_frames()[0]
    nba_games = games[games['TEAM_ID'].isin([team['id'] for team in nba_teams])]

    # Calculate the stats for each team
    team_stats = []
    for team in nba_teams:
        team_id = team['id']
        team_games = nba_games[(nba_games['TEAM_ID'] == team_id)]
        wins = len(team_games[team_games['WL'] == 'W'])
        losses = len(team_games[team_games['WL'] == 'L'])
        win_percentage = wins / (wins + losses) if (wins + losses) > 0 else 0

        points = team_games['PTS'].sum()
        games_played = len(team_games)
        average_points = points / games_played if games_played > 0 else 0
        fg3_pct = team_games['FG3_PCT'].sum() / games_played if games_played > 0 else 0
        avg_reb = team_games['REB'].sum() / games_played if games_played > 0 else 0
        avg_oreb = team_games['OREB'].sum() / games_played if games_played > 0 else 0
        avg_ast = team_games['AST'].sum() / games_played if games_played > 0 else 0
        avg_tov =team_games['TOV'].sum() / games_played if games_played > 0 else 0

        team_stat = {
            'TEAM_ID': team_id,
            'team_name': team['full_name'],
            'games_played': games_played,
            'wins': wins,
            'losses': losses,
            'win_percentage': win_percentage,
            'average_points': average_points,
            'fg3_pct': fg3_pct,
            'reb':avg_reb,
            'oreb':avg_oreb,
            'ast':avg_ast,
            'tov':avg_tov
        }
        team_stats.append(team_stat)

    return pd.DataFrame(team_stats)

def preprocess_data(odds_data, team_stats_data):
    # Merge the data, drop unnecessary columns, and create features
    #merged_data = pd.merge(odds_data, team_stats_data, left_on='team1_id', right_on='TEAM_ID')
    #merged_data = pd.merge(merged_data, team_stats_data, left_on='team2_id', right_on='TEAM_ID', suffixes=('_team1', '_team2'))
    #merged_data = pd.merge(odds_data, team_stats_data)
    #merged_data = pd.merge(merged_data, team_stats_data)
    
    for team in odds_data['team1_name']:
        merged_df = odds_data.merge(team_stats_data, left_on="team1_name", right_on="team_name", suffixes=('', '_team1'))
        merged_df = merged_df.merge(team_stats_data, left_on="team2_name", right_on="team_name", suffixes=('_team1', '_team2'))
        merged_df = merged_df.drop(columns=['TEAM_ID_team1', 'team_name_team1', 'TEAM_ID_team2', 'team_name_team2'])
    # Add any additional preprocessing steps needed
    return merged_df

# Create new features
def create_new_features(data):
    #data['team1_points_per_game'] = data['team1_points'] / data['team1_games_played']
    #data['team2_points_per_game'] = data['team2_points'] / data['team2_games_played']
    return data

def select_features(data):
    # Customize this function to select the features you want to use
    features_to_drop = ['game_id', 'team1_name', 'team2_name']
    return data.drop(columns=features_to_drop)

# Train multiple models and combine them using a VotingClassifier
def train_stacked_model(X_train, y_train):
    rf = RandomForestClassifier()
    lr = LogisticRegression()
    xgb = XGBClassifier()

    model = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('xgb', xgb)],
        voting='soft'
    )
    model.fit(X_train, y_train)
    return model

def train_model(X_train, y_train):
    # Perform hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return predictions, probabilities

def predict_upcoming_games(model, upcoming_odds_data, team_stats_data):
    # Preprocess data
    upcoming_data = preprocess_data(upcoming_odds_data, team_stats_data)

    # Create new features
    upcoming_data = create_new_features(upcoming_data)

    # Select features
    X_upcoming = select_features(upcoming_data)

    # Make predictions
    predictions, probabilities = make_predictions(model, X_upcoming)

    # Add predictions to the DataFrame
    upcoming_data['predicted_winner'] = np.where(predictions == 1, upcoming_data['team1_name'], upcoming_data['team2_name'])
    upcoming_data['team1_probability'] = probabilities[:, 1]
    upcoming_data['team2_probability'] = probabilities[:, 0]

    return upcoming_data


# Main code
if __name__ == "__main__":
    # Define custom date range for fetching odds data
    #start_date = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')
    #end_date = datetime.today().strftime('%Y-%m-%d')

    start_date = '%2021-%10-%19'
    end_date = '%2022-%04-%10'

    # Fetch data
    odds_data = fetch_odds_data(start_date, end_date, API_KEY)
    team_stats_data = fetch_team_stats_data()
    team_ids = team_stats_data['TEAM_ID'].unique()
    # Preprocess data
    data = preprocess_data(odds_data, team_stats_data)

    # Create new features
    data = create_new_features(data)

    # Select features
    X = select_features(data)
    y = np.where(data['team1_odds'] < data['team2_odds'], 1, 0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the stacked model
    model = train_stacked_model(X_train, y_train)

    # Make predictions
    predictions, probabilities = make_predictions(model, X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion matrix:\n", cm)


'''
    # Fetch upcoming games odds data (change the start_date and end_date to the desired range)
    upcoming_start_date = '2023-04-01'
    upcoming_end_date = '2023-04-10'
    upcoming_odds_data = fetch_odds_data(upcoming_start_date, upcoming_end_date, API_KEY)

    # Predict upcoming games
    upcoming_game_predictions = predict_upcoming_games(model, upcoming_odds_data, team_stats_data)
    print("Upcoming game predictions:\n", upcoming_game_predictions)
'''  