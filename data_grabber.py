# © Anthony Krivonos 2019
# NBA ML
# data_grabber.py
# March 3rd, 2019

import csv
import requests
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamelog

# Get list of games
games = leaguegamelog.LeagueGameLog(direction='ASC', league_id='00', player_or_team_abbreviation='T', season_all_time='2018-19', season_type_all_star='Regular Season', sorter='PTS')

# Get data and headers 
dataDict = games.league_game_log.get_dict()
data, headers = dataDict['data'], dataDict['headers']

# List of NBA teams
nba_teams = teams.get_teams()

##
#
#   Write Games List to CSV
#
##

# Store games in CSV
with open('data/gameslist.csv', mode='w') as games_file:
    games_writer = csv.writer(games_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Write header
    games_writer.writerow(headers)
    # Write rows
    for data_row in data:
        games_writer.writerow(data_row)

##
#
#   Create Training Data
#
##

# --------------------------------------
# Features:
# --------------------------------------
# x0: Points (PTS)
# x1: Field Goal Percentage: (FG_PCT)
# x2: Three-Point Percentage: (FG3_PCT)
# x3: Free Throw Percentage: (FT_PCT)
# x4: Rebounds (REB)
# x5: Assists (AST)
# x6: Steals (STL)
# x7: Blocks (BLK)
# x8: Turnovers (TOV)
# --------------------------------------
# Class:
# --------------------------------------
# y: 0 (lose), 1 (win)
# --------------------------------------

# List of training headers
training_headers = [
    'MIN',
    'FGM',
    'FGA',
    'FG_PCT',
    'FG3M',
    'FG3A',
    'FG3_PCT',
    'FTM',
    'FTA',
    'FT_PCT',
    'OREB',
    'DREB',
    'REB',
    'AST',
    'STL',
    'BLK',
    'TOV',
    'PF',
    'PTS',
    'PLUS_MINUS',
    'WL'            # Make sure WL is last
]
training_indices = {}
for header in training_headers:
    training_indices[header] = headers.index(header)

# Store training data in CSV
with open('data/trainingdata.csv', mode='w') as training_file:
    training_writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Write header
    training_writer.writerow(training_headers)
    # Write rows
    for data_row in data:
        train_row = []
        for header in training_headers:
            index = training_indices[header]
            if (header == 'WL'):
                # -1 for loss, 1 for win
                raw = data_row[index]
                train_row.append(1 if raw == 'W' else -1)
            else:
                train_row.append(data_row[index])
        training_writer.writerow(train_row)