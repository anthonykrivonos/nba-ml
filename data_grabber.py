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
    'PTS',
    'FG_PCT',
    'FG3_PCT',
    'FT_PCT',
    'REB',
    'AST',
    'STL',
    'BLK',
    'TOV',
    'WL'
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
        train_row = [
            data_row[training_indices['PTS']],
            data_row[training_indices['FG_PCT']],
            data_row[training_indices['FG3_PCT']],
            data_row[training_indices['FT_PCT']],
            data_row[training_indices['REB']],
            data_row[training_indices['AST']],
            data_row[training_indices['STL']],
            data_row[training_indices['BLK']],
            data_row[training_indices['TOV']],
            # Compute 1 for win, 0 for loss
            (1 if data_row[training_indices['WL']] == 'W' else 0)
        ]
        training_writer.writerow(train_row)