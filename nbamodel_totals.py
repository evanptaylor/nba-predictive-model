import requests
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamelog
import config

API_KEY = config.api_key

def fetch_odds_data(api_key):
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"

    # Add parameters to a dictionary
    params = {
        'api_key': api_key,
        'regions': 'us',
        'markets': 'totals',
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'bookmakers': ['bovada'],
        #'bookmakers': 1
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception(f"Failed to fetch odds data. Error: {response.text}")
    odds_list = []
    for game in data:
        if game is None:
            continue
        if not game['bookmakers']:
            continue

        bookmaker = game['bookmakers'][0]
        if not bookmaker['markets']:
            continue

        market = bookmaker['markets'][0]
        if not market['outcomes']:
            continue

        outcomes = market['outcomes']
        game_data = {
            
            'game_id': game['id'],
            'team1_name': game['home_team'],
            'team2_name': game['away_team'],
            'over_odds': outcomes[0]['price'],
            'under_odds': outcomes[1]['price'],
            'total_points': outcomes[0]['point']
        }
        odds_list.append(game_data)
    return pd.DataFrame(odds_list)


def fetch_team_stats_data(start_date, end_date):
    nba_teams = teams.get_teams()

    # Fetch NBA game logs for all time
    #game_log = leaguegamelog.LeagueGameLog(date_from_nullable=start_date, date_to_nullable=end_date, league_id='00')
    game_log = leaguegamelog.LeagueGameLog(season='2022-23', league_id='00')
    games = game_log.get_data_frames()[0]
    nba_games = games[games['TEAM_ID'].isin([team['id'] for team in nba_teams])]
    #games_21_22 = games[games.SEASON_ID.str[-4:] == '2021']

    #print(nba_games)
    # Calculate the stats for each team
    team_stats = []
    for team in nba_teams:
        team_id = team['id']
        team_games = nba_games[(nba_games['TEAM_ID'] == team_id)]
        #wonlast = 1 if team_games[team_games['WL'][len(team_games[team_games['WL']])-1] == 'W'] else 0
        wins = len(team_games[team_games['WL'] == 'W'])
        losses = len(team_games[team_games['WL'] == 'L'])
        win_percentage = wins / (wins + losses) if (wins + losses) > 0 else 0

        points = team_games['PTS'].sum()
        games_played = len(team_games)
        average_points = points / games_played if games_played > 0 else 0
        fg3_pct = team_games['FG3_PCT'].sum() / games_played if games_played > 0 else 0
        fg2_pct = team_games['FG_PCT'].sum() / games_played if games_played > 0 else 0
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
            'fg2_pct': fg2_pct,
            'reb':avg_reb,
            'oreb':avg_oreb,
            'ast':avg_ast,
            'tov':avg_tov,
        }
        team_stats.append(team_stat)
    return pd.DataFrame(team_stats)

def preprocess_data(odds_data, team_stats_data):
    merged_df = odds_data.merge(team_stats_data, left_on="team1_name", right_on="team_name", suffixes=('', '_team1'))
    merged_df = merged_df.merge(team_stats_data, left_on="team2_name", right_on="team_name", suffixes=('_team1', '_team2'))
    merged_df = merged_df.drop(columns=['TEAM_ID_team1', 'team_name_team1', 'TEAM_ID_team2', 'team_name_team2'])

    return merged_df


# Create new features
def create_new_features(data):
    #data['team1_points_per_game'] = data['team1_points'] / data['team1_games_played']
    #data['team2_points_per_game'] = data['team2_points'] / data['team2_games_played']
    return data

def select_features(data):
    # Customize this function to select the features you want to use
    features_to_drop = ['game_id', 'team1_name', 'team2_name','over_odds','under_odds']
    return data.drop(columns=features_to_drop)

# Train multiple models and combine them using a VotingClassifier
def train_stacked_model(X_train, y_train):
    rf = RandomForestClassifier()
    lr = LogisticRegression(max_iter=3000)
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
    upcoming_data['predicted_outcome'] = np.where(predictions == 1, "over", "under")
    upcoming_data['over_probability'] = probabilities[:, 1]
    upcoming_data['under_probability'] = probabilities[:, 0]


    return upcoming_data

def format_excel(file):
    # Load the workbook and select the sheet
    wb = load_workbook(file)
    ws = wb.active

    # Define the fill colors for the winner and loser
    winner_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
    loser_fill = PatternFill(start_color="FFC0CB", end_color="FFC0CB", fill_type="solid")

    # Iterate through the rows and apply the fill colors
    for row in range(2, ws.max_row + 1):
        team1_prob = ws.cell(row=row, column=6).value
        team2_prob = ws.cell(row=row, column=7).value
        
        if team1_prob > team2_prob:
            ws.cell(row=row, column=1).fill = winner_fill
            ws.cell(row=row, column=2).fill = loser_fill
            ws.cell(row=row, column=3).fill = winner_fill
            ws.cell(row=row, column=4).fill = loser_fill
            ws.cell(row=row, column=5).fill = winner_fill
            ws.cell(row=row, column=6).fill = winner_fill
            ws.cell(row=row, column=7).fill = loser_fill
        else:
            ws.cell(row=row, column=1).fill = loser_fill
            ws.cell(row=row, column=2).fill = winner_fill
            ws.cell(row=row, column=3).fill = loser_fill
            ws.cell(row=row, column=4).fill = winner_fill
            ws.cell(row=row, column=5).fill = winner_fill
            ws.cell(row=row, column=6).fill = loser_fill
            ws.cell(row=row, column=7).fill = winner_fill
    
    for col in range(1, 26):
        ws.column_dimensions[chr(col + 64)].width = 20
    # Save the formatted workbook
    wb.save("upcoming_bets.xlsx")

# Main code
if __name__ == "__main__":
    # Define custom date range for fetching training data

    start_date = "%2022-%10-%19"
    end_date = "%2023-%03-%30"

    # Fetch data
    odds_data = fetch_odds_data(API_KEY)
    #odss_data = process_over_under_odds(odds_data)
    team_stats_data = fetch_team_stats_data(start_date, end_date)
    team_ids = team_stats_data['TEAM_ID'].unique()
    # Preprocess data
    data = preprocess_data(odds_data, team_stats_data)
    # Create new features
    data = create_new_features(data)
    # Select features
    X = select_features(data)
    print(X)
    y = np.where(data['total_points'] == 'over', 1, 0)

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
    #cm = confusion_matrix(y_test, predictions)
    #print("Confusion matrix:\n", cm)

    # Fetch upcoming games odds data (change the start_date and end_date to the desired range)
    upcoming_odds_data = fetch_odds_data(API_KEY)

    # Predict upcoming games
    upcoming_game_predictions = predict_upcoming_games(model, upcoming_odds_data, team_stats_data)
    
    keep = ['team1_name', 'team2_name', 'over', 'under', 'total_points', 'predicted_outcome', 'over_probability', 'under_probability']
    upcoming_game_predictions = upcoming_game_predictions.filter(keep)
    
    #print("Upcoming game predictions:\n", upcoming_game_predictions)
    
    upcoming_game_predictions.to_excel("predictions.xlsx", index=False)
    format_excel("predictions.xlsx")
    print("Done! Open predictions.xlsx to view")
    

