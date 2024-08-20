#!/usr/bin/env python
# coding: utf-8

# This file will be used to scrape team data from basketball-reference.com from 1980-2024.

# import libraries
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import numpy as np
import html5lib

# get data from 2023-24 season as an example
url = 'https://www.basketball-reference.com/leagues/NBA_2024.html'

# make the request
result = requests.get(url = url)

# parse the html source using BeautifulSoup
soup = bs(result.text, features = 'lxml')

# grab the per game statistics table
per_game_table = str(soup.find_all('table', id = 'per_game-team'))

# grab the advanced statistics table
advanced_stats_table = str(soup.find_all('table', id = 'advanced-team'))

# convert both to pandas data frames
per_game_df = pd.read_html(per_game_table, encoding = 'utf-8')[0].iloc[:-1]
advanced_stats_df = pd.read_html(advanced_stats_table, encoding = 'utf-8')[0].iloc[:-1]

# reformat the columns of the advanced_stats dataframe
advanced_stats_df.columns = advanced_stats_df.columns.droplevel()

# merge the two data frames on the Team column
season_2024_df = pd.merge(left = per_game_df, right = advanced_stats_df, how = 'inner', on = 'Team')

# null values are in the unnamed columns
season_2024_df.isnull().sum()

# we will drop these columns
season_2024_df = season_2024_df.drop(['Unnamed: 17_level_1', 'Unnamed: 22_level_1', 'Unnamed: 27_level_1'], axis = 1)

# add the season year to the dataframe
season_2024_df['Season'] = [2024]*len(season_2024_df)

season_2024_df.head()



### THE ABOVE WILL BE THE WORKFLOW IN THE FOR LOOP AS WE ITERATE THROUGH YEARS
### FOR LOOP WILL BE SPLIT IN TWO DUE TO HIGH VOLUME OF API REQUESTS, TO BE RUN AT DIFFERENT TIMES

# set up a list to store the data frames
seasons = []

# grab data for seasons from 1980-2002, since 1979-80 was the season the NBA added the 3 point line
for year in range(1980, 2003, 1):
    # set the appropriate url
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}.html'
    # make the request
    result = requests.get(url = url)
    # parse the html source
    soup = bs(result.text, features = 'lxml')
    
    # find the tables from the page and convert them to data frames, omitting average row
    per_game_table = str(soup.find_all('table', id = 'per_game-team'))
    advanced_stats_table = str(soup.find_all('table', id = 'advanced-team'))
    
    per_game_df = pd.read_html(per_game_table, encoding = 'utf-8')[0].iloc[:-1]
    advanced_stats_df = pd.read_html(advanced_stats_table, encoding = 'utf-8')[0].iloc[:-1]
    
    # reformat the columns of the advanced_stats dataframe
    advanced_stats_df.columns = advanced_stats_df.columns.droplevel()
    # merge the two data frames on the Team column
    season_df = pd.merge(left = per_game_df, right = advanced_stats_df, how = 'inner', on = 'Team')
    # drop the columns with null values because of formatting
    season_df = season_df.drop(['Unnamed: 17_level_1', 'Unnamed: 22_level_1', 'Unnamed: 27_level_1'], axis = 1)
    # add the season year to the data frame
    season_df['Year'] = [year]*len(season_df)

    # add the data frame to the seasons list
    seasons.append(season_df)

# now grab data for seasons from 2003 to 2024
for year in range(2003, 2025, 1):
    # set the appropriate url
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}.html'
    # make the request
    result = requests.get(url = url)
    # parse the html source
    soup = bs(result.text, features = 'lxml')
    
    # find the tables from the page and convert them to data frames, omitting average row
    per_game_table = str(soup.find_all('table', id = 'per_game-team'))
    advanced_stats_table = str(soup.find_all('table', id = 'advanced-team'))
    
    per_game_df = pd.read_html(per_game_table, encoding = 'utf-8')[0].iloc[:-1]
    advanced_stats_df = pd.read_html(advanced_stats_table, encoding = 'utf-8')[0].iloc[:-1]
    
    # reformat the columns of the advanced_stats dataframe
    advanced_stats_df.columns = advanced_stats_df.columns.droplevel()
    # merge the two data frames on the Team column
    season_df = pd.merge(left = per_game_df, right = advanced_stats_df, how = 'inner', on = 'Team')
    # drop the columns with null values because of formatting
    season_df = season_df.drop(['Unnamed: 17_level_1', 'Unnamed: 22_level_1', 'Unnamed: 27_level_1'], axis = 1)
    # add the season year to the data frame
    season_df['Year'] = [year]*len(season_df)

    # add the data frame to the seasons list
    seasons.append(season_df)

# concatenate all the seasons into one data frame
team_stats_df = pd.concat(seasons)

# save the data frame as a .csv file
team_stats_df.to_csv('nba_team_stats.csv', index = False)

