#!/usr/bin/env python
# coding: utf-8

# This file will be used to perform an extensive exploratory data analysis on the cleaned NBA team data

# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
sns.set_theme(style = 'white', palette = 'colorblind')

# load in the cleaned data
nba_df = pd.read_csv('cleaned_nba_team_stats.csv')

nba_df.head()

# examine the correlations between the numeric variables; many high correlations
plt.figure(figsize = (15,12), dpi = 300)
heat_map = sns.heatmap(data = nba_df.corr(numeric_only = True), cmap = 'vlag')
plt.title('Correlation Map of the Numeric Variables', fontsize = 22)
fig = heat_map.get_figure()
fig.savefig('correlation_heatmap.png', dpi = 300, bbox_inches = 'tight');

# examine a pairplot of select numeric variables
plt.figure(figsize = (9,9), dpi = 300)
sns.pairplot(data = nba_df[['3-Point_Attempt_Rate',
                                        'Offensive_Rating',
                                        'Defensive_Rating',
                                        'Average_Age',
                                        'Win_Percentage',
                                        'Made_Playoffs']],
                         hue = 'Made_Playoffs', palette = ['#ff6961', 'skyblue'])
plt.suptitle('Pairplot of Select Numeric Variables', y = 1.02)
plt.savefig('pairplot.png', dpi = 300, bbox_inches = 'tight');


# First, we will look at how the NBA has changed over time

### EXPLORE THE THREE-POINT SHOT, INTRODUCED IN THE FIRST SEASON IN OUR DATA SET (1980)

# examine the average attempt rate per season
fig = plt.figure(dpi = 300)
sns.lineplot(data = nba_df, x = 'Season', y = '3-Point_Attempt_Rate', estimator = 'mean', color = 'blue')
plt.title('Average 3-Point Attempt Rate Per Season (1980-2024)')
plt.ylabel('Average 3-Point Attempt Rate')
fig.savefig('three_point_attempt_trend.png', dpi = 300, bbox_inches = 'tight');

# examine the average number of points scored per game by season
fig = plt.figure(dpi = 300)
sns.lineplot(data = nba_df, x = 'Season', y = 'Points', estimator = 'mean', color = 'blue')
plt.title('Average PPG Per Season (1980-2024)')
plt.ylabel('Average Points Per Game')
fig.savefig('points_trend.png', dpi = 300, bbox_inches = 'tight');

# let's pull in the century for visualization
nba_df['Century'] = nba_df['Season'].apply(lambda x: np.where(x < 2000, '20th Century', '21st Century'))

# examine the relationship between 3-point attempts and total points scored by century
fig = plt.figure(figsize = (6,4), dpi = 300)
sns.scatterplot(data = nba_df, x = '3-Pointers_Attempted', y = 'Points', hue = 'Century',
               alpha = 0.7, linewidth = 0, palette = ['#ff6961', 'skyblue'])
plt.title('Relationship Between 3-Point Attempts and Points Scored by Century')
plt.xlabel('Average 3-Point Attempts Per Game')
plt.ylabel('Average Points Per Game')
fig.savefig('threes_and_points_by_century.png', dpi = 300, bbox_inches = 'tight');

# see how the pace of the game has changed over time
fig = plt.figure(dpi = 300)
sns.lineplot(data = nba_df, x = 'Season', y = 'Pace', color = 'blue')
plt.title('Change in Pace of Game Over Time (1980-2024)')
plt.ylabel('Average Game Pace')
fig.savefig('pace_over_time.png', dpi = 300, bbox_inches = 'tight');


# Next, we will move to examine the profile of teams with winning records

# surprisingly, defensive rating appears negatively correlated with win percentage
fig = plt.figure(dpi = 300)
ax = sns.regplot(data = nba_df, x = 'Defensive_Rating', y = 'Win_Percentage', ci = None, color = 'skyblue')
plt.title('Win Percentage Decreases when Defensive Rating Increases')
plt.xlabel('Defensive Rating')
plt.ylabel('Win Percentage')
fig.savefig('defensive_rating_win_percentage.png', dpi = 300, bbox_inches = 'tight');

# however, win percentage increases when offensive rating increases
fig = plt.figure(dpi = 300)
sns.regplot(data = nba_df, x = 'Offensive_Rating', y = 'Win_Percentage', color = 'skyblue', ci = None)
plt.title('Win Percentage Increases when Offensive Rating Increases')
plt.xlabel('Offensive Rating')
plt.ylabel('Win Percentage')
fig.savefig('offensive_rating_win_percentage.png', dpi = 300, bbox_inches = 'tight');

# interestingly, older teams tend to have better records
fig = plt.figure(dpi = 300)
sns.regplot(data = nba_df, x = 'Average_Age', y = 'Win_Percentage', color = 'skyblue', ci = None)
plt.title('Older Teams Tend to Have Higher Win Percentages')
plt.xlabel('Average Age')
plt.ylabel('Win Percentage')
fig.savefig('age_win_percentage.png', dpi = 300, bbox_inches = 'tight');


# Now we will visually build a profile of championship teams

# visualize the correlation of each numeric variable with winning the title in a heatmap
fig = plt.figure(figsize = (6,9), dpi = 300)
sns.heatmap(data = pd.DataFrame(nba_df.corr(numeric_only = True)['Won_Title'].sort_values(ascending = False)[1:]),
            cmap = 'vlag', annot = True)
plt.title('Correlation of Numeric Variables with Winning Championships')
fig.savefig('correlations_with_titles.png', dpi = 300, bbox_inches = 'tight');

# find proportion of championship teams that also had the highest net rating
100*len(nba_df[(nba_df['Won_Title']==1) & (nba_df['Highest_Net_Rating']==1)]) / len(nba_df[nba_df['Won_Title']==1])

# examine the distribution of defensive ratings for championship teams versus other teams
fig = plt.figure(figsize = (6,4), dpi = 300)
sns.boxplot(data = nba_df, x = 'Defensive_Rating', hue = 'Won_Title', palette = ['skyblue', '#ff6961'])
plt.title('Defense Does NOT Win Championships')
plt.xlabel('Defensive Rating')
fig.savefig('defense_and_championships.png', dpi = 300, bbox_inches = 'tight');

# examine the distribution of offensive ratings for championship teams versus other teams
fig = plt.figure(figsize = (6,4), dpi = 300)
sns.boxplot(data = nba_df, x = 'Offensive_Rating', hue = 'Won_Title', palette = ['skyblue', '#ff6961'])
plt.title('Offense Wins Championships')
plt.xlabel('Offensive Rating')
fig.savefig('offense_and_championships.png', dpi = 300, bbox_inches = 'tight');

# examine the 5 most dominant teams in the last 45 years by margin of victory
dominant_indices = list(nba_df['Margin_of_Victory'].sort_values(ascending = False)[:5].index)
fig = plt.figure(dpi = 300)
bar_plot = sns.barplot(data = nba_df.iloc[dominant_indices], x = nba_df.iloc[dominant_indices].index,
                       y = 'Margin_of_Victory', order = [393,1014,1225,423,984], color = 'skyblue')
bar_plot.set_xticks(labels = ["'96 Bulls", "'17 Warriors", "'24 Celtics", "'97 Bulls", "'16 Warriors"],
                    ticks = bar_plot.get_xticks(), rotation = 45)
plt.title('Five Most Dominant Teams by Avg Margin of Victory (1980-2024)')
plt.xlabel('Team')
plt.ylabel('Average Margine of Victory')
fig.savefig('dominant_teams.png', dpi = 300, bbox_inches = 'tight');