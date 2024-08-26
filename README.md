
# NBA: Statistics and Success

This is an EDA and machine learning project in Python designed to understand trends in the NBA and what best predicts success.

## TLDR

- The 3-point shot has radically changed the game and is a key component of any successful offense.
- Both in terms of win percentage and titles, the modern NBA selects for high-powered offenses much more than stingy defenses.
- Winning the turnover battle is critical to championship success.
## Highlights

### Defense Paradox

<img src="https://github.com/seanmurphy21/NBA_Win_Percentages/blob/main/Plots/defense_and_championships.png?raw=true" width="500" height="390" />

<img src="https://github.com/seanmurphy21/NBA_Win_Percentages/blob/main/Plots/defensive_rating_win_percentage.png?raw=true" width="500" height="400" />

- Surprisingly, teams with higher defensive ratings tend to win fewer games and championships on the whole

### Three-Point Shot

<img src="https://github.com/seanmurphy21/NBA_Win_Percentages/blob/main/Plots/three_point_attempt_trend.png?raw=true" width="500" height="390" />

<img src="https://github.com/seanmurphy21/NBA_Win_Percentages/blob/main/Plots/threes_and_points_by_century.png?raw=true" width="500" height="370" />

- Threes have been more heavily integrated into offenses over time, but only began to increase scoring at the turn of the 21st Century.

### Traditional vs. Advanced Stats

<img src="https://github.com/seanmurphy21/NBA_Win_Percentages/blob/main/Plots/enet_partial_dependence.png?raw=true" height="390" />

- Margin of victory, net rating, and age are most important advanced predictors of win percentage.

<img src="https://github.com/seanmurphy21/NBA_Win_Percentages/blob/main/Plots/svr_partial_dependence.png?raw=true" width="500" height="370" />

- Turnovers, field goal attempts, and steals are among the most important win percentage predictors in terms of traditional statistics.




## Full Project Summary

View the full project summary slides here: [Full Project Summary](https://github.com/seanmurphy21/NBA_Win_Percentages/releases/tag/v1)
