import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from nba_api.stats.endpoints import leaguegamefinder, teamdashboardbylastngames, playerdashboardbylastngames

# Replace with your API key
API_KEY = "YOUR_API_KEY"

def fetch_odds_data(start_date, end_date):
    url = f"https://api.the-odds-api.com/v4/odds/?apiKey={API_KEY}&sport=basketball_nba&region=us&mkt=h2h&dateFormat=iso&start={start_date}&end={end_date}"
    response = requests.get(url)
    data = response.json()
    
    odds_list = []
    for game in data:
        game_data = {
            'game_id': game['id'],
            'team1_id': game['teams'][0]['id'],
            'team2_id': game['teams'][1]['id'],
            'team1_odds': game['odds'][0]['h2h'][0],
            'team2_odds': game['odds'][0]['h2h'][1]
        }
        odds_list.append(game_data)
    
    return pd.DataFrame(odds_list)

def fetch_team_stats_data():
    gamefinder = leaguegamefinder.LeagueGameFinder()
    games = gamefinder.get_data_frames()[0]
    
    # Filter for NBA games (League ID 00 is for NBA)
    nba_games = games[games['LEAGUE_ID'] == '00']
    team_ids = nba_games['TEAM_ID'].unique()

    # Fetch team statistics for the last N games
    team_stats = []
    for team_id in team_ids:
        dashboard = teamdashboardbylastngames.TeamDashboardByLastNGames(team_id=team_id, last_n_games=10)
        stats = dashboard.get_data_frames()[1].iloc[0]
        team_stats.append(stats)

    return pd.DataFrame(team_stats)

def fetch_player_stats_data(team_ids):
    player_stats = []

    for team_id in team_ids:
        players = playerdashboardbylastngames.PlayerDashboardByLastNGames(team_id=team_id, last_n_games=10)
        player_stats.extend(players.get_data_frames()[1].to_dict('records'))

    return pd.DataFrame(player_stats)

def preprocess_data(odds_data, team_stats_data):
    # Merge the data, drop unnecessary columns, and create features
    merged_data = pd.merge(odds_data, team_stats_data, left_on='team1_id', right_on='TEAM_ID')
    merged_data = pd.merge(merged_data, team_stats_data, left_on='team2_id', right_on='TEAM_ID', suffixes=('_team1', '_team2'))

    # Add any additional preprocessing steps needed
    return merged_data

# Create new features
def create_new_features(data):
    data['team1_points_per_game'] = data['team1_points'] / data['team1_games_played']
    data['team2_points_per_game'] = data['team2_points'] / data['team2_games_played']
    return data

def select_features(data):
    # Customize this function to select the features you want to use
    features_to_drop = ['game_id', 'team1_id', 'team2_id', 'team1_odds', 'team2_odds']
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

# Main code
if __name__ == "__main__":
    # Define custom date range for fetching odds data
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Fetch data
    odds_data = fetch_odds_data(start_date, end_date)
    team_stats_data = fetch_team_stats_data()
    team_ids = team_stats_data['TEAM_ID'].unique()
    player_stats_data = fetch_player_stats_data(team_ids)

    # Preprocess data
    data = preprocess_data(odds_data, team_stats_data, player_stats_data)

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
