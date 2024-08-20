#!/usr/bin/env python
# coding: utf-8

# This file will be used to clean the data from scrape.py and prepare it for analysis

# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

# load in the data
nba_df = pd.read_csv('nba_team_stats.csv')

nba_df.head()

# first, we will replace column names with more readable versions based on the basketball-reference.com glossary
new_cols = ['Rank',
            'Team',
            'Games_Played',
            'Minutes_Played',
            'Field_Goals_Made',
            'Field_Goals_Attempted',
            'Field_Goal_Percentage',
            '3-Pointers_Made',
            '3-Pointers_Attempted',
            '3-Point_Field_Goal_Percentage',
            '2-Pointers_Made',
            '2-Pointers_Attempted',
            '2-Point_Field_Goal_Percentage',
            'Free_Throws_Made',
            'Free_Throws_Attempted',
            'Free_Throw_Percentage',
            'Offensive_Rebounds',
            'Defensive_Rebounds',
            'Total_Rebounds',
            'Assists',
            'Steals',
            'Blocks',
            'Turnovers',
            'Personal_Fouls',
            'Points',
            'Rank2',
            'Average_Age',
            'Wins',
            'Losses',
            'Expected_Wins',
            'Expected_Losses',
            'Margin_of_Victory',
            'Strength_of_Schedule',
            'Simple_Rating_System',
            'Offensive_Rating',
            'Defensive_Rating',
            'Net_Rating',
            'Pace',
            'Free_Throw_Attempt_Rate',
            '3-Point_Attempt_Rate',
            'True_Shooting_Percentage',
            'Offensive_Effective_Field_Goal_Percentage',
            'Offensive_Turnover_Percentage',
            'Offensive_Rebound_Percentage',
            'Offensive_Free_Throws_Per_Field_Goal_Attempt',
            'Opponent_Effective_Field_Goal_Percentage',
            'Opponent_Turnover_Percentage',
            'Defensive_Rebound_Percentage',
            'Opponent_Free_Throws_Per_Field_Goal_Attempt',
            'Arena',
            'Total_Home_Attendance',
            'Attendance_Per_Game',
            'Season']

# change the column names to the more readable versions above
nba_df.columns = new_cols

# first, we will drop the two rank columns which are superfluous to our purposes
nba_df = nba_df.drop(['Rank', 'Rank2'], axis = 1)

# we will create a playoffs column based on the team names with an asterisk (in playoffs)
nba_df['Made_Playoffs'] = nba_df['Team'].apply(lambda x: np.where('*' in x, 1, 0))

# then we will remove the asterisk from the team names
nba_df['Team'] = nba_df['Team'].apply(lambda x: x.strip('*'))

# define a function to calculate win percentage
def win_percentage_calculator(wins, games_played):
    return np.round(wins/games_played, decimals = 3)

# add the win percentage column to the dataframe
nba_df['Win_Percentage'] = nba_df.apply(lambda x: win_percentage_calculator(x.Wins, x.Games_Played), axis = 1)

# create a dictionary that stores the NBA champion for each season https://www.basketball-reference.com/playoffs/
nba_champ_dict = {1980: 'Los Angeles Lakers',
                  1981: 'Boston Celtics',
                  1982: 'Los Angeles Lakers',
                  1983: 'Philadelphia 76ers',
                  1984: 'Boston Celtics',
                  1985: 'Los Angeles Lakers',
                  1986: 'Boston Celtics',
                  1987: 'Los Angeles Lakers',
                  1988: 'Los Angeles Lakers',
                  1989: 'Detroit Pistons',
                  1990: 'Detroit Pistons',
                  1991: 'Chicago Bulls',
                  1992: 'Chicago Bulls',
                  1993: 'Chicago Bulls',
                  1994: 'Houston Rockets',
                  1995: 'Houston Rockets',
                  1996: 'Chicago Bulls',
                  1997: 'Chicago Bulls',
                  1998: 'Chicago Bulls',
                  1999: 'San Antonio Spurs',
                  2000: 'Los Angeles Lakers',
                  2001: 'Los Angeles Lakers',
                  2002: 'Los Angeles Lakers',
                  2003: 'San Antonio Spurs',
                  2004: 'Detroit Pistons',
                  2005: 'San Antonio Spurs',
                  2006: 'Miami Heat',
                  2007: 'San Antonio Spurs',
                  2008: 'Boston Celtics',
                  2009: 'Los Angeles Lakers',
                  2010: 'Los Angeles Lakers',
                  2011: 'Dallas Mavericks',
                  2012: 'Miami Heat',
                  2013: 'Miami Heat',
                  2014: 'San Antonio Spurs',
                  2015: 'Golden State Warriors',
                  2016: 'Cleveland Cavaliers',
                  2017: 'Golden State Warriors',
                  2018: 'Golden State Warriors',
                  2019: 'Toronto Raptors',
                  2020: 'Los Angeles Lakers',
                  2021: 'Milwaukee Bucks',
                  2022: 'Golden State Warriors',
                  2023: 'Denver Nuggets',
                  2024: 'Boston Celtics'}

# define a function to indicate whether a team was the NBA champion for a particular season
def was_champion(season, team):
    if nba_champ_dict[season] == team:
        return 1
    else:
        return 0

# add the column indicating whether the team was the title winner for that particular season
nba_df['Won_Title'] = nba_df.apply(lambda x: was_champion(x.Season, x.Team), axis = 1)

# view the average number of games played each season
plt.figure(figsize = (12,8), dpi = 200)
sns.barplot(data = nba_df, x = 'Season', y = 'Games_Played', estimator = 'mean', errorbar = None)
plt.xticks(rotation = 45)
plt.title('Average Number of Games Played Per Season');

# outliers are lockouts in '99 and '12, COVID in '20 and '21
# since games played is consistent and we have win %, drop Games_Played, Wins, and Losses
nba_df = nba_df.drop(['Games_Played', 'Wins', 'Losses'], axis = 1)

# since each arena is unique for each team, the info will be redundant; drop the column
nba_df = nba_df.drop('Arena', axis = 1)

# we could fill the total home attendance based on average attendance and number of home games
# however, since some seasons had fewer total games, average attendance will be better to use
# drop the Total_Home_Attendance column
nba_df = nba_df.drop('Total_Home_Attendance', axis = 1)

# let's take a look at the Minutes_Played column for the 2024 season
plt.figure()
sns.barplot(data = nba_df[nba_df['Season']==2024], x = 'Team', y = 'Minutes_Played')
plt.title('Minutes Played Per Game for Each 2024 Team')
plt.xticks(rotation = 90);

# since there will be very little variation to this column, we will remove it
nba_df = nba_df.drop('Minutes_Played', axis = 1)

# we will also remove expected wins and expected losses, since these essentially what we will try to predict
nba_df = nba_df.drop(['Expected_Wins', 'Expected_Losses'], axis = 1)

# now check for missing values in each column
nba_df.isnull().sum()

# see where the Attendance_Per_Game values are missing
nba_df[nba_df['Attendance_Per_Game'].isnull()][['Team','Season','Attendance_Per_Game']]



### SEE HOW ATTENDANCE PER GAME IS RELATED TO WIN PERCENTAGE IN EACH SEASON

# grab all the seasons
seasons = list(range(1980,2025,1))
# set up storage for the correlations
corrs = []

for season in range(1980,2025,1):
    correlation = nba_df[nba_df['Season']==season].corr(numeric_only = True)['Attendance_Per_Game'].loc['Win_Percentage']
    corrs.append(correlation)

# plot out the results of the above
plt.figure(figsize = (8,6), dpi = 200)
sns.lineplot(x = seasons, y = corrs)
plt.xlabel('Season')
plt.ylabel('Correlation')
plt.title('Correlation Between Attendance Per Game and Win Percentage for 1980-2024 Seasons');

# the two may be related, although the relationship varies a lot season to season
# let us examine the attendance during the 2021 season
plt.figure(figsize = (12,6), dpi = 200)
sns.barplot(data = nba_df[nba_df['Season']==2021], x = 'Team', y = 'Attendance_Per_Game')
plt.title('Attendance per Game in the 2021 NBA Season')
plt.xticks(rotation = 90);

# we see above a wide variability in attendance, with many null values, difficult to estimate missing values
# we could either remove the 2021 season and seek to estimate remaining missing vals or remove the attendance column
# let's see what teams would be missing values if we removed the 2021 season
nba_df[(nba_df['Season']!=2021) & (nba_df['Attendance_Per_Game'].isnull())][['Team', 'Season', 'Attendance_Per_Game']]

# SuperSonics are missing this value for many seasons in a row, making it hard to accurately estimate
# since we don't want to lose 2021 season data, we will simply remove the Attendance_Per_Game column
nba_df = nba_df.drop('Attendance_Per_Game', axis = 1)

# add a column indicating whether or not a team finished above .500
nba_df['Winning_Record'] = nba_df['Win_Percentage'].apply(lambda x: np.where(x>.5, 1, 0))

# create a dictionary with the highest net-ranked team in each season
highest_net_dict = {}
for season in range(1980,2025,1):
    # grab the index location of the highest ranked team
    highest_index = nba_df[nba_df['Season']==season]['Net_Rating'].idxmax()
    # grab the team associated with that index location
    highest_team = nba_df.iloc[highest_index]['Team']
    # add the team and the year to the dictionary
    highest_net_dict[season] = highest_team

# define a function to indicate whether a team had the highest net rating for a particular season
def had_highest_net_rating(season, team):
    if highest_net_dict[season] == team:
        return 1
    else:
        return 0
    
# add the column indicating whether the team had the highest net rating for that particular season
nba_df['Highest_Net_Rating'] = nba_df.apply(lambda x: had_highest_net_rating(x.Season, x.Team), axis = 1)

# data is ready for analysis, export the data
nba_df.to_csv('cleaned_nba_team_stats.csv', index = False)

