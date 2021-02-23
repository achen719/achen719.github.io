---
title: "A Naive Attempt at Predicting Postseason Success"
layout: single
classes: wide

---
The repository for the original notebook can be found [here](https://github.com/achen719/Predicting-Playoff-Performance).

The following will be an exploration on how to properly quantify postseason performance in the NBA, how the characteristics of a team's regular season performance translates to the postseason, and an initial (truthfully, naive) foray into modeling postseason success based on regular season data.

I had wanted to attempt a relatively simple approach for predicting the playoffs by using cumulative team-level features from the regular season. This turned out to be an extremely optimistic approach. The models essentially tell us that good regular season teams perform well and less good regular season teams perform worse, but without especially high accuracy. Also, quite frankly, we do not need a model to tell us that.

NBA basketball in and of itself is incredibly complex, but playoff basketball is even moreso and turns out to be quite different from what occurs in the regular season. Part of this is simply because teams play fewer games and a more biased sample of games, so variation is bound to occur. Another reason is that teams are no longer facing multiple teams over a short period of time and instead, can hone in and gameplan for one team.

However, while the actual modeling proves to be faulty, I do believe the initial data analysis to be useful and more importantly, gives us a direction in terms of understanding what is necessary to model for the postseason.

The contents of the notebook are as follows:
1. **Quantifying Regular Season and Playoff Performance**: we explore ways to contextualize and adjust regular season and playoff performance metrics for the state of the league they occur in.
2. **How the Regular Season Translates to the Playoffs**: we look at how regular season features hold up in the playoffs and then which of those features are correlated with postseason success.
3. **Modeling for Postseason Performance with Hyperparameter Tuning**: we test how well linear regression models, support vector machines, random forest regression models, and gradient-boosted decision tree models predict postseason success.
4. **What Went Wrong and What to Change**: we explain possible changes in approach to improve our performance.


```python
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import math
from scipy.stats import skew
```


```python
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
```


```python
conn = create_engine('mysql+pymysql://{user}:{pw}@localhost/{db}'.format(user='root',pw='password',db='nba_db'))
cursor = conn.connect()
```


```python
sql_statement = 'select * from boxscores_team'
df_base = pd.read_sql(sql_statement, conn)
```

The data was scraped from the official NBA stats webpage and consists of team-level boxscores -- summary statistics of a team's performance in a game -- from the 1997-98 NBA season to the 2020-21 NBA season. A copy of the unwrangled data has been supplied in the repository.


```python
df = df_base.copy()
```

#### Creating, deleting, and aggregating variables

The data was parsed from play-by-play data and for the purposes of this notebook contains far too many unnecessay features. Below, we manipulate the data for our purposes.


```python
df['MIN_PLAYED'] = df['MIN_PLAYED']/600
# Create an indicator for wins
df['WIN'] = 0
df['WIN'].loc[df['PTS'] > df['OPP_PTS']] = 1
# Create a # of games column
df['GAME'] = 1
# Include opponent
df['OPPONENT'] = df['MATCHUP'].str.slice(stop = 3)
df['TEAM_HELPER'] = df['MATCHUP'].str.slice(start = 6, stop = 9)
df['OPPONENT'].loc[df['TEAM_ABBREVIATION'] != df['TEAM_HELPER']] = df['TEAM_HELPER']
# These features are not particularly relevant on a team-level
cols_to_drop = ['FG_BLKD', 'FG3_BLKD', 'BAD_PASS_TOV', 'OFF_FOUL', 
              'FG_0_2_ASTD', 'FG_3_9_ASTD', 'FG_10_15_ASTD', 'FG_16_3PT_ASTD', 'FG_CORNER_3_ASTD', 
              'FG_24_26_ASTD', 'FG_27_30_ASTD', 'FG_31_35_ASTD', 'FG_36_ASTD', 
              'NONSHOOT_FT', 'NONSHOOT_FTA',
              'AST_0_2', 'AST_3_9', 'AST_10_15', 'AST_16_3PT','AST_CORNER_3', 
              'AST_24_26', 'AST_27_30', 'AST_31_35', 'AST_36']
cols_to_drop += ['OPP_' + col for col in cols_to_drop]
# Aggregate non-rim 2PT shots
fg = ['FG_3_9', 'FG_10_15', 'FG_16_3PT']
fga = ['FGA_3_9', 'FGA_10_15', 'FGA_16_3PT']
df['FG_NONRIM_2'] = df[fg].sum(axis = 1)
df['FGA_NONRIM_2'] = df[fga].sum(axis = 1)
df['OPP_FG_NONRIM_2'] = df[['OPP_' + col for col in fg]].sum(axis = 1)
df['OPP_FGA_NONRIM_2'] = df[['OPP_' + col for col in fga]].sum(axis = 1)
# Aggregate 3s from 31+ Feet
fg_3 = ['FG_31_35', 'FG_36']
fga_3 = ['FGA_31_35', 'FGA_36']
df['FG_31'] = df[fg_3].sum(axis = 1)
df['FGA_31'] = df[fga_3].sum(axis = 1)
df['OPP_FG_31'] = df[['OPP_' + col for col in fg_3]].sum(axis = 1)
df['OPP_FGA_31'] = df[['OPP_' + col for col in fga_3]].sum(axis = 1)
# True shooting attempts
df['TSA'] = df['FGA'] + df['FG2_FTA']/2 + df['FG3_FTA']/3
df['OPP_TSA'] = df['OPP_FGA'] + df['OPP_FG2_FTA']/2 + df['OPP_FG3_FTA']/3
# Dropping all unnecessary columns
cols_to_drop += fg + fga + ['OPP_' + col for col in fg + fga] + fg_3 + fga_3 + ['OPP_' + col for col in fg_3 + fga_3]
df.drop(['TEAM_HELPER'] + cols_to_drop, axis = 1, inplace = True)
```


```python
df.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GAME_ID</th>
      <th>GAME_DATE</th>
      <th>SEASON_YEAR</th>
      <th>SEASON_TYPE</th>
      <th>TEAM_ID</th>
      <th>TEAM_ABBREVIATION</th>
      <th>HOME_OR_AWAY</th>
      <th>MATCHUP</th>
      <th>MIN_PLAYED</th>
      <th>PTS</th>
      <th>FGA</th>
      <th>FG</th>
      <th>FTA</th>
      <th>FT</th>
      <th>FG3A</th>
      <th>FG3</th>
      <th>FG_ASTD</th>
      <th>FG3_ASTD</th>
      <th>REB</th>
      <th>OREB</th>
      <th>DREB</th>
      <th>TOV</th>
      <th>LIVE_TOV</th>
      <th>FG_0_2</th>
      <th>FG_CORNER_3</th>
      <th>FG_24_26</th>
      <th>FG_27_30</th>
      <th>FGA_0_2</th>
      <th>FGA_CORNER_3</th>
      <th>FGA_24_26</th>
      <th>FGA_27_30</th>
      <th>FG2_FT</th>
      <th>FG2_AND1_FT</th>
      <th>FG3_FT</th>
      <th>FG3_AND1_FT</th>
      <th>FG2_FTA</th>
      <th>FG2_AND1_FTA</th>
      <th>FG3_FTA</th>
      <th>FG3_AND1_FTA</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>OFF_POSS</th>
      <th>DEF_POSS</th>
      <th>OPP_PTS</th>
      <th>OPP_FGA</th>
      <th>OPP_FG</th>
      <th>OPP_FTA</th>
      <th>OPP_FT</th>
      <th>OPP_FG3A</th>
      <th>OPP_FG3</th>
      <th>OPP_FG_ASTD</th>
      <th>OPP_FG3_ASTD</th>
      <th>OPP_REB</th>
      <th>OPP_OREB</th>
      <th>OPP_DREB</th>
      <th>OPP_TOV</th>
      <th>OPP_LIVE_TOV</th>
      <th>OPP_FG_0_2</th>
      <th>OPP_FG_CORNER_3</th>
      <th>OPP_FG_24_26</th>
      <th>OPP_FG_27_30</th>
      <th>OPP_FGA_0_2</th>
      <th>OPP_FGA_CORNER_3</th>
      <th>OPP_FGA_24_26</th>
      <th>OPP_FGA_27_30</th>
      <th>OPP_FG2_FT</th>
      <th>OPP_FG2_AND1_FT</th>
      <th>OPP_FG3_FT</th>
      <th>OPP_FG3_AND1_FT</th>
      <th>OPP_FG2_FTA</th>
      <th>OPP_FG2_AND1_FTA</th>
      <th>OPP_FG3_FTA</th>
      <th>OPP_FG3_AND1_FTA</th>
      <th>OPP_AST</th>
      <th>OPP_STL</th>
      <th>OPP_BLK</th>
      <th>WIN</th>
      <th>GAME</th>
      <th>OPPONENT</th>
      <th>FG_NONRIM_2</th>
      <th>FGA_NONRIM_2</th>
      <th>OPP_FG_NONRIM_2</th>
      <th>OPP_FGA_NONRIM_2</th>
      <th>FG_31</th>
      <th>FGA_31</th>
      <th>OPP_FG_31</th>
      <th>OPP_FGA_31</th>
      <th>TSA</th>
      <th>OPP_TSA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0021901318</td>
      <td>2020-08-14</td>
      <td>2019-20</td>
      <td>Regular+Season</td>
      <td>1610612743</td>
      <td>DEN</td>
      <td>AWAY</td>
      <td>DEN - TOR</td>
      <td>48.0</td>
      <td>109.0</td>
      <td>87.0</td>
      <td>36.0</td>
      <td>23.0</td>
      <td>21.0</td>
      <td>38.0</td>
      <td>16.0</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>41.0</td>
      <td>9.0</td>
      <td>32.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>25.0</td>
      <td>7.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>117.0</td>
      <td>90.0</td>
      <td>45.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>35.0</td>
      <td>18.0</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>51.0</td>
      <td>13.0</td>
      <td>38.0</td>
      <td>19.0</td>
      <td>9.0</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>2.0</td>
      <td>24.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>1</td>
      <td>TOR</td>
      <td>7.0</td>
      <td>24.0</td>
      <td>8.0</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>95.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0021901318</td>
      <td>2020-08-14</td>
      <td>2019-20</td>
      <td>Regular+Season</td>
      <td>1610612761</td>
      <td>TOR</td>
      <td>HOME</td>
      <td>DEN - TOR</td>
      <td>48.0</td>
      <td>117.0</td>
      <td>90.0</td>
      <td>45.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>35.0</td>
      <td>18.0</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>51.0</td>
      <td>13.0</td>
      <td>38.0</td>
      <td>19.0</td>
      <td>9.0</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>32.0</td>
      <td>2.0</td>
      <td>24.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>109.0</td>
      <td>87.0</td>
      <td>36.0</td>
      <td>23.0</td>
      <td>21.0</td>
      <td>38.0</td>
      <td>16.0</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>41.0</td>
      <td>9.0</td>
      <td>32.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>25.0</td>
      <td>7.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>1</td>
      <td>1</td>
      <td>DEN</td>
      <td>8.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>95.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0021901317</td>
      <td>2020-08-14</td>
      <td>2019-20</td>
      <td>Regular+Season</td>
      <td>1610612760</td>
      <td>OKC</td>
      <td>AWAY</td>
      <td>LAC - OKC</td>
      <td>53.0</td>
      <td>103.0</td>
      <td>106.0</td>
      <td>38.0</td>
      <td>22.0</td>
      <td>13.0</td>
      <td>44.0</td>
      <td>14.0</td>
      <td>18.0</td>
      <td>10.0</td>
      <td>48.0</td>
      <td>11.0</td>
      <td>37.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>30.0</td>
      <td>4.0</td>
      <td>30.0</td>
      <td>7.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>115.0</td>
      <td>113.0</td>
      <td>107.0</td>
      <td>85.0</td>
      <td>34.0</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>37.0</td>
      <td>12.0</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>61.0</td>
      <td>12.0</td>
      <td>49.0</td>
      <td>20.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>24.0</td>
      <td>8.0</td>
      <td>25.0</td>
      <td>4.0</td>
      <td>26.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>LAC</td>
      <td>8.0</td>
      <td>32.0</td>
      <td>9.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>115.0</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0021901317</td>
      <td>2020-08-14</td>
      <td>2019-20</td>
      <td>Regular+Season</td>
      <td>1610612746</td>
      <td>LAC</td>
      <td>HOME</td>
      <td>LAC - OKC</td>
      <td>53.0</td>
      <td>107.0</td>
      <td>85.0</td>
      <td>34.0</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>37.0</td>
      <td>12.0</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>61.0</td>
      <td>12.0</td>
      <td>49.0</td>
      <td>20.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>24.0</td>
      <td>8.0</td>
      <td>25.0</td>
      <td>4.0</td>
      <td>26.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>113.0</td>
      <td>115.0</td>
      <td>103.0</td>
      <td>106.0</td>
      <td>38.0</td>
      <td>22.0</td>
      <td>13.0</td>
      <td>44.0</td>
      <td>14.0</td>
      <td>18.0</td>
      <td>10.0</td>
      <td>48.0</td>
      <td>11.0</td>
      <td>37.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>30.0</td>
      <td>4.0</td>
      <td>30.0</td>
      <td>7.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1</td>
      <td>1</td>
      <td>OKC</td>
      <td>9.0</td>
      <td>24.0</td>
      <td>8.0</td>
      <td>32.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>104.0</td>
      <td>115.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0021901315</td>
      <td>2020-08-14</td>
      <td>2019-20</td>
      <td>Regular+Season</td>
      <td>1610612755</td>
      <td>PHI</td>
      <td>AWAY</td>
      <td>HOU - PHI</td>
      <td>48.0</td>
      <td>134.0</td>
      <td>87.0</td>
      <td>49.0</td>
      <td>22.0</td>
      <td>18.0</td>
      <td>38.0</td>
      <td>18.0</td>
      <td>31.0</td>
      <td>13.0</td>
      <td>51.0</td>
      <td>8.0</td>
      <td>43.0</td>
      <td>16.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>101.0</td>
      <td>102.0</td>
      <td>96.0</td>
      <td>80.0</td>
      <td>35.0</td>
      <td>16.0</td>
      <td>14.0</td>
      <td>48.0</td>
      <td>12.0</td>
      <td>25.0</td>
      <td>11.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>17.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>7.0</td>
      <td>23.0</td>
      <td>18.0</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>1</td>
      <td>1</td>
      <td>HOU</td>
      <td>18.0</td>
      <td>35.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>97.0</td>
      <td>87.0</td>
    </tr>
  </tbody>
</table>

### Converting the Data into Rates

Our very first step is simple: prorate all metrics to 100 possessions and account for the difference in value of 2-point shots and 3-point shots by calculating effective field goal percentage.


```python
def calc_fields(df):
    """
    For calculating percentages and rates.
    """
    df['WIN%'] = df['WIN']/df['GAME']
    # Offensive Rating / Defensive Rating
    df['PACE'] = 48 * (df['OFF_POSS']/df['MIN_PLAYED'])
    df['ORTG'] = 100 * (df['PTS']/df['OFF_POSS'])
    df['DRTG'] = 100 * (df['OPP_PTS']/df['DEF_POSS'])
    df['NRTG'] = df['ORTG'] - df['DRTG']
    # Rebounding Rates
    df['OREB_RATE'] = 100 * (df['OREB']/df['OFF_POSS'])
    df['DREB_RATE'] = 100 * (df['DREB']/df['DEF_POSS'])
    df['DREB%'] = df['DREB']/(df['DREB'] + df['OPP_OREB'])
    # Assist Rates
    df['AST_RATE'] = 100 * (df['AST']/df['OFF_POSS'])
    df['%ASTD'] = df['FG_ASTD']/df['FG']
    # Steals and Blocks
    df['STL_RATE'] = 100 * (df['STL']/df['DEF_POSS'])
    df['BLK_RATE'] = 100 * (df['BLK']/df['DEF_POSS'])
    shot_locs = ['_0_2', '_NONRIM_2', '_CORNER_3', '_24_26', '_27_30', '_31']
    prefixes = ['', 'OPP_']
    for prefix in prefixes:
        # Per 100 POSS RATES
        if prefix == '':
            poss = 'OFF_POSS'
        else:
            poss = 'DEF_POSS'
        # Shot Locations
        for shot_loc in shot_locs:
            if shot_loc in ['_0_2', '_NONRIM_2']:
                multiplier = 1
            else:
                multiplier = 1.5
            # FG & FGA Rates
            df[prefix + 'FGA' + shot_loc + '_RATE'] = 100 * (df[prefix + 'FGA' + shot_loc] / df[poss])
            # eFG%
            df[prefix + 'eFG' + shot_loc + '%'] = multiplier * df[prefix + 'FG' + shot_loc] / df[prefix + 'FGA' + shot_loc]
        # Three-pointers/free throw attempts
        df[prefix + 'FG3A_RATE'] = 100 * df[prefix + 'FG3A']/df[poss]
        df[prefix + 'FTA_RATE'] = 100 * df[prefix + 'FTA']/df[poss]
        # Shooting Percentages
        df[prefix + 'TS%'] = df[prefix + 'PTS']/(2 * df[prefix + 'TSA'])
        df[prefix + 'eFG%'] = (df[prefix + 'FG'] + 0.5 * df[prefix + 'FG3'])/df[prefix + 'FGA']
        df[prefix + 'FG3%'] = df[prefix + 'FG3']/df[prefix + 'FG3A']
        df[prefix + 'FT%'] = df[prefix + 'FT']/df[prefix + 'FTA']
        # Turnover
        df[prefix + 'TOV_RATE'] = 100 * (df[prefix + 'TOV']/df[poss])
        df[prefix + 'LIVE_TOV_RATE'] = 100 * (df[prefix + 'LIVE_TOV']/df[poss])
```


```python
# Columns to be dropped after each groupby aggregation
cols_drop_1 = ['MIN_PLAYED', 'OFF_POSS', 'DEF_POSS', 'GAME']
cols_drop_2 = ['PTS', 'FGA', 'FG', 'FTA', 'FT', 'FG3A', 'FG3', 'FG_ASTD','FG3_ASTD',
                'REB', 'OREB', 'DREB', 'TOV', 'LIVE_TOV', 'AST', 'STL', 'BLK',
                'FG_0_2', 'FG_NONRIM_2', 'FG_CORNER_3', 'FG_24_26', 'FG_27_30', 'FG_31',
                'FGA_0_2', 'FGA_NONRIM_2', 'FGA_CORNER_3', 'FGA_24_26', 'FGA_27_30', 'FGA_31',
                'FG2_FT', 'FG2_AND1_FT', 'FG3_FT', 'FG3_AND1_FT', 
                'FG2_FTA', 'FG2_AND1_FTA', 'FG3_FTA', 'FG3_AND1_FTA', 'TSA']
cols_drop_3 = ['OPP_' + col for col in cols_drop_2]
```

## 1. Quantifying Regular Season and Playoff Performance
The first thing we must establish is how we evaluate or define performance. To do so, however, we must first understand how to contextualize performances of teams from season to season.
### The Not-so-Steady State of the League
A big issue with regards to standardizing or scaling our data is that the state of the league is in constant flux. Players change teams, players simply get better or worse, rule changes alter style of play, and so on. To demonstrate, we look at the average points scored per game and the average three-pointers attempted per game, prorated to 100 possessions, since the 1997-98 season. 

Note that PO stands for Playoffs and RS for Regular Season.

### Adjusting Regular Season Data to League Average

If we standardize our data using the entire dataset then we fail to account for the heteregeneity of each season. Therefore, we deal with the reality of an everchanging league by comparing features to the **league average of that season**. 


```python
# Drop extra columns in League Averages dataframe
df_lg.drop(cols_drop_1 + cols_drop_2 + cols_drop_3, axis = 1, inplace = True)
# Team-level performance aggregated to each regular season and playoffs
df_teams = df.groupby(['TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON_YEAR', 'SEASON_TYPE']).sum()
df_teams.reset_index(inplace = True)
calc_fields(df_teams)
df_teams.drop(cols_drop_1 + cols_drop_2 + cols_drop_3, axis = 1, inplace = True)
# Regular Season data
df_rs = df_teams.loc[df_teams['SEASON_TYPE'] == 'Regular+Season']
# Merge with Regular Season data
df_rs_lg_adj = pd.merge(df_rs, df_lg, 
                         how = 'left', 
                         on = ['SEASON_YEAR', 'SEASON_TYPE'],
                         suffixes = ['', '_LG_AVG']).sort_values(['SEASON_YEAR', 'SEASON_TYPE'])
```
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/52.embed" height="525" width="100%"></iframe>

```python
def league_avg_adj(df):
    """
    Finds the difference between each feature and the league average for the season
    """
    cols = df.select_dtypes(include = [np.number]).columns.tolist()
    cols_to_adj = [col for col in cols if col[-7:] != '_LG_AVG']
    cols_lg = [col for col in cols if col[-7:] == '_LG_AVG']
    for col in cols_to_adj:
        if col + '_LG_AVG' in cols_lg:
            df['r' + col] = df[col] - df[col + '_LG_AVG']
```


```python
league_avg_adj(df_rs_lg_adj)
```

### Adjusting Playoff Results to the Quality of Opposition

In the postseason, the sample of games a team plays is not only much smaller (teams play at most 28 games) than that of the regular season, but the quality of competition is both stronger and more biased (teams play at most 4 different teams). The results of the postseason are incredibly matchup-dependent and therefore, we require a way to incorporate quality of competition in our performance metrics.

To handle this, we compare performance of a team in the postseason **relative to their opponent's regular season performance**. This will both give us a more accurate view of how well a team has performed and stabilizes the postseason data.

More formally, we define the response variables in the following manner:
* **Relative Offensive Rating in the Postseason (rORTG_PO)** as the difference between offensive rating against an opponent and the opponent's regular season defensive rating.
* **Relative Defensive Rating in the Postseason (rDRTG_PO)** as the difference between defensive rating against an opponent and the opponent's regular season offensive rating. (The lower/more negative the better.)
* **Relative Net Rating (rNRTG_PO)** as the difference between relative offensive and defensive rating.


```python
# Aggregation of Team-level metrics to a Matchup-level
df_po = df.loc[df['SEASON_TYPE'] == 'Playoffs']
df_po = df_po.groupby(['TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON_YEAR', 'SEASON_TYPE', 'OPPONENT']).sum()
df_po.reset_index(inplace = True)
calc_fields(df_po)
# Merge playoff data with opponent's regular season data
df_po_matchup_unagg = pd.merge(df_po, df_rs[['SEASON_YEAR', 'TEAM_ABBREVIATION', 'ORTG', 'DRTG', 'NRTG']], how = 'left', 
                         left_on = ['SEASON_YEAR', 'OPPONENT'], right_on = ['SEASON_YEAR', 'TEAM_ABBREVIATION'],
                         suffixes = ['', '_OPP'])
# Relative ratings for each matchup
df_po_matchup_unagg['rORTG'] = df_po_matchup_unagg['ORTG'] - df_po_matchup_unagg['DRTG_OPP']
df_po_matchup_unagg['rDRTG'] = df_po_matchup_unagg['DRTG'] - df_po_matchup_unagg['ORTG_OPP']
df_po_matchup_unagg['rNRTG'] = df_po_matchup_unagg['rORTG'] - df_po_matchup_unagg['rDRTG']

# Total Weights
weight = df_po_matchup_unagg[['SEASON_YEAR', 'TEAM_ABBREVIATION', 'OFF_POSS', 'DEF_POSS']].groupby(['SEASON_YEAR', 'TEAM_ABBREVIATION'], as_index = False).sum()
df_po_matchup = pd.merge(df_po_matchup_unagg, weight, how = 'left', on = ['SEASON_YEAR', 'TEAM_ABBREVIATION'], suffixes = ['', '_TOT'])
df_po_matchup['rORTG'] = df_po_matchup['rORTG'] * df_po_matchup['OFF_POSS']/df_po_matchup['OFF_POSS_TOT']
df_po_matchup['rDRTG'] = df_po_matchup['rDRTG'] * df_po_matchup['DEF_POSS']/df_po_matchup['DEF_POSS_TOT']
df_po_matchup['rNRTG'] = df_po_matchup['rORTG'] - df_po_matchup['rDRTG']
# Drop matchup-related columns
cols_to_drop = [col for col in df_po_matchup.columns if col[-4:] == '_OPP'] + ['OFF_POSS_TOT', 'DEF_POSS_TOT']
df_po_matchup.drop(cols_to_drop, axis = 1, inplace = True)
# Create calculated fields and drop any unnecessary columns
df_matchup_adj = df_po_matchup.groupby(['SEASON_YEAR', 'TEAM_ABBREVIATION', 'SEASON_TYPE']).sum()
df_matchup_adj.reset_index(inplace = True)
calc_fields(df_matchup_adj)

df_matchup_adj.drop(cols_drop_1 + cols_drop_2 + cols_drop_3, axis = 1, inplace = True)
df_matchup_adj.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEASON_YEAR</th>
      <th>TEAM_ABBREVIATION</th>
      <th>SEASON_TYPE</th>
      <th>WIN</th>
      <th>WIN%</th>
      <th>PACE</th>
      <th>ORTG</th>
      <th>DRTG</th>
      <th>NRTG</th>
      <th>OREB_RATE</th>
      <th>DREB_RATE</th>
      <th>DREB%</th>
      <th>AST_RATE</th>
      <th>%ASTD</th>
      <th>STL_RATE</th>
      <th>BLK_RATE</th>
      <th>FGA_0_2_RATE</th>
      <th>eFG_0_2%</th>
      <th>FGA_NONRIM_2_RATE</th>
      <th>eFG_NONRIM_2%</th>
      <th>FGA_CORNER_3_RATE</th>
      <th>eFG_CORNER_3%</th>
      <th>FGA_24_26_RATE</th>
      <th>eFG_24_26%</th>
      <th>FGA_27_30_RATE</th>
      <th>eFG_27_30%</th>
      <th>FGA_31_RATE</th>
      <th>eFG_31%</th>
      <th>FG3A_RATE</th>
      <th>FTA_RATE</th>
      <th>TS%</th>
      <th>eFG%</th>
      <th>FG3%</th>
      <th>FT%</th>
      <th>TOV_RATE</th>
      <th>LIVE_TOV_RATE</th>
      <th>OPP_FGA_0_2_RATE</th>
      <th>OPP_eFG_0_2%</th>
      <th>OPP_FGA_NONRIM_2_RATE</th>
      <th>OPP_eFG_NONRIM_2%</th>
      <th>OPP_FGA_CORNER_3_RATE</th>
      <th>OPP_eFG_CORNER_3%</th>
      <th>OPP_FGA_24_26_RATE</th>
      <th>OPP_eFG_24_26%</th>
      <th>OPP_FGA_27_30_RATE</th>
      <th>OPP_eFG_27_30%</th>
      <th>OPP_FGA_31_RATE</th>
      <th>OPP_eFG_31%</th>
      <th>OPP_FG3A_RATE</th>
      <th>OPP_FTA_RATE</th>
      <th>OPP_TS%</th>
      <th>OPP_eFG%</th>
      <th>OPP_FG3%</th>
      <th>OPP_FT%</th>
      <th>OPP_TOV_RATE</th>
      <th>OPP_LIVE_TOV_RATE</th>
      <th>rORTG</th>
      <th>rDRTG</th>
      <th>rNRTG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-98</td>
      <td>ATL</td>
      <td>Playoffs</td>
      <td>1</td>
      <td>0.250000</td>
      <td>86.500000</td>
      <td>101.156069</td>
      <td>100.291545</td>
      <td>0.864524</td>
      <td>13.872832</td>
      <td>32.653061</td>
      <td>0.767123</td>
      <td>19.653179</td>
      <td>0.519084</td>
      <td>4.664723</td>
      <td>5.539359</td>
      <td>22.543353</td>
      <td>0.641026</td>
      <td>41.040462</td>
      <td>0.380282</td>
      <td>0.289017</td>
      <td>0.000000</td>
      <td>19.364162</td>
      <td>0.604478</td>
      <td>0.289017</td>
      <td>0.000000</td>
      <td>0.867052</td>
      <td>0.0</td>
      <td>20.809249</td>
      <td>25.722543</td>
      <td>0.522388</td>
      <td>0.494863</td>
      <td>0.375000</td>
      <td>0.685393</td>
      <td>14.161850</td>
      <td>9.826590</td>
      <td>23.906706</td>
      <td>0.646341</td>
      <td>48.688047</td>
      <td>0.425150</td>
      <td>1.166181</td>
      <td>1.125000</td>
      <td>8.163265</td>
      <td>0.589286</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.874636</td>
      <td>0.0</td>
      <td>10.204082</td>
      <td>25.947522</td>
      <td>0.532508</td>
      <td>0.510563</td>
      <td>0.400000</td>
      <td>0.606742</td>
      <td>14.577259</td>
      <td>10.787172</td>
      <td>-1.367715</td>
      <td>-4.941553</td>
      <td>3.573839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-98</td>
      <td>CHH</td>
      <td>Playoffs</td>
      <td>4</td>
      <td>0.444444</td>
      <td>84.777778</td>
      <td>97.640891</td>
      <td>103.363519</td>
      <td>-5.722628</td>
      <td>11.009174</td>
      <td>31.177232</td>
      <td>0.678873</td>
      <td>25.032765</td>
      <td>0.640940</td>
      <td>2.846054</td>
      <td>3.234153</td>
      <td>24.377457</td>
      <td>0.564516</td>
      <td>48.492792</td>
      <td>0.435135</td>
      <td>1.179554</td>
      <td>0.833333</td>
      <td>11.009174</td>
      <td>0.482143</td>
      <td>0.262123</td>
      <td>0.000000</td>
      <td>0.655308</td>
      <td>0.0</td>
      <td>13.106160</td>
      <td>22.411533</td>
      <td>0.509576</td>
      <td>0.478659</td>
      <td>0.320000</td>
      <td>0.684211</td>
      <td>15.203145</td>
      <td>11.664482</td>
      <td>25.614489</td>
      <td>0.621212</td>
      <td>44.501940</td>
      <td>0.389535</td>
      <td>0.905563</td>
      <td>0.642857</td>
      <td>14.489004</td>
      <td>0.549107</td>
      <td>1.164295</td>
      <td>0.166667</td>
      <td>0.776197</td>
      <td>0.0</td>
      <td>17.335058</td>
      <td>26.520052</td>
      <td>0.518831</td>
      <td>0.480030</td>
      <td>0.335821</td>
      <td>0.731707</td>
      <td>13.454075</td>
      <td>9.443726</td>
      <td>-2.191148</td>
      <td>-2.650596</td>
      <td>0.459448</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-98</td>
      <td>CHI</td>
      <td>Playoffs</td>
      <td>15</td>
      <td>0.714286</td>
      <td>86.286837</td>
      <td>106.885246</td>
      <td>100.611791</td>
      <td>6.273455</td>
      <td>16.994536</td>
      <td>30.533927</td>
      <td>0.709302</td>
      <td>23.551913</td>
      <td>0.601116</td>
      <td>6.729700</td>
      <td>5.283648</td>
      <td>26.830601</td>
      <td>0.613035</td>
      <td>46.338798</td>
      <td>0.386792</td>
      <td>1.038251</td>
      <td>0.631579</td>
      <td>12.349727</td>
      <td>0.511062</td>
      <td>0.928962</td>
      <td>0.264706</td>
      <td>0.546448</td>
      <td>0.0</td>
      <td>14.863388</td>
      <td>32.295082</td>
      <td>0.522436</td>
      <td>0.472377</td>
      <td>0.323529</td>
      <td>0.734349</td>
      <td>13.497268</td>
      <td>9.344262</td>
      <td>29.365962</td>
      <td>0.553030</td>
      <td>41.212458</td>
      <td>0.423752</td>
      <td>1.612903</td>
      <td>0.568966</td>
      <td>10.734149</td>
      <td>0.497409</td>
      <td>0.723026</td>
      <td>0.230769</td>
      <td>0.611791</td>
      <td>0.0</td>
      <td>13.681869</td>
      <td>27.141268</td>
      <td>0.521626</td>
      <td>0.476238</td>
      <td>0.313008</td>
      <td>0.750000</td>
      <td>16.629588</td>
      <td>12.625139</td>
      <td>5.189052</td>
      <td>-6.449337</td>
      <td>11.638389</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-98</td>
      <td>CLE</td>
      <td>Playoffs</td>
      <td>1</td>
      <td>0.250000</td>
      <td>84.250000</td>
      <td>95.845697</td>
      <td>105.029586</td>
      <td>-9.183888</td>
      <td>12.166172</td>
      <td>28.698225</td>
      <td>0.723881</td>
      <td>21.958457</td>
      <td>0.691589</td>
      <td>7.100592</td>
      <td>3.550296</td>
      <td>22.255193</td>
      <td>0.640000</td>
      <td>43.620178</td>
      <td>0.353741</td>
      <td>0.890208</td>
      <td>0.500000</td>
      <td>5.637982</td>
      <td>0.473684</td>
      <td>0.296736</td>
      <td>0.000000</td>
      <td>0.296736</td>
      <td>0.0</td>
      <td>7.121662</td>
      <td>41.839763</td>
      <td>0.520968</td>
      <td>0.449187</td>
      <td>0.291667</td>
      <td>0.723404</td>
      <td>20.474777</td>
      <td>13.649852</td>
      <td>19.526627</td>
      <td>0.727273</td>
      <td>44.674556</td>
      <td>0.403974</td>
      <td>2.958580</td>
      <td>0.750000</td>
      <td>13.017751</td>
      <td>0.545455</td>
      <td>0.591716</td>
      <td>0.750000</td>
      <td>0.887574</td>
      <td>0.0</td>
      <td>17.455621</td>
      <td>26.923077</td>
      <td>0.567093</td>
      <td>0.514493</td>
      <td>0.372881</td>
      <td>0.780220</td>
      <td>16.568047</td>
      <td>11.834320</td>
      <td>-3.722279</td>
      <td>-1.103316</td>
      <td>-2.618963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-98</td>
      <td>HOU</td>
      <td>Playoffs</td>
      <td>2</td>
      <td>0.400000</td>
      <td>89.000000</td>
      <td>95.056180</td>
      <td>102.237136</td>
      <td>-7.180957</td>
      <td>13.483146</td>
      <td>34.004474</td>
      <td>0.737864</td>
      <td>18.202247</td>
      <td>0.558621</td>
      <td>3.131991</td>
      <td>5.592841</td>
      <td>19.775281</td>
      <td>0.545455</td>
      <td>44.044944</td>
      <td>0.372449</td>
      <td>1.797753</td>
      <td>0.187500</td>
      <td>18.202247</td>
      <td>0.425926</td>
      <td>0.224719</td>
      <td>0.000000</td>
      <td>0.674157</td>
      <td>0.0</td>
      <td>20.898876</td>
      <td>33.483146</td>
      <td>0.479592</td>
      <td>0.416446</td>
      <td>0.258065</td>
      <td>0.731544</td>
      <td>14.157303</td>
      <td>11.460674</td>
      <td>26.621924</td>
      <td>0.621849</td>
      <td>45.413870</td>
      <td>0.379310</td>
      <td>2.237136</td>
      <td>0.450000</td>
      <td>7.382550</td>
      <td>0.636364</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.447427</td>
      <td>0.0</td>
      <td>10.067114</td>
      <td>33.557047</td>
      <td>0.528935</td>
      <td>0.479620</td>
      <td>0.377778</td>
      <td>0.693333</td>
      <td>14.541387</td>
      <td>10.961969</td>
      <td>-7.349916</td>
      <td>-8.062136</td>
      <td>0.712220</td>
    </tr>
  </tbody>
</table>




We now compare the two methods of evaluating playoff performance: unadjusted offensive and defensive ratings vs. matchup-adjusted offensive and defensive ratings.

#### How to interpret the scatterplots

* The *size* of the bubbles represent the number of wins and the *color* represents the normalized unadjusted and matchup-adjusted point differentials. 
* For offensive rating, the higher the better and for defensive rating, the lower the better, which means that the further from the dotted $y = x$ line the higher the team's net rating.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/55.embed" height="525" width="100%"></iframe> 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/57.embed" height="525" width="100%"></iframe>

#### Key Takeaways
* When scrolling through the seasons in the **unadjusted** scatterplot, one should notice that the data points are slowly shifting towards the right, mirroring the league-wide trend in scoring.
* In the **matchup-adjusted** scatterplot, the data points are much more stable, when accounting for season-to-season variance.

Using **matchup-adjusted** offensive and defensive ratings gives us a much closer approximation of *actual* playoff performance, but a key component of it -- the opponents each team faces -- is not fully known, which we will see rears its head when we start modeling.

## 2. How the Regular Season Translates to the Playoffs
One of the biggest challenge to  predicting postseason performance is the fact that play in the regular season and the postseason occur under two different environments. In the regular season, teams are tasked with preparing for many different teams and a hectic travel schedule while in the postseason, teams are allowed to hone in one team at a time. This places even further emphasis on how matchup-reliant playoff results are.

Below we look at how top-level offensive, defensive, and overall ratings translate to the playoffs.
### How to interpret the scatterplots
* First, note that it is better to have a higher **rORTG_RS** or Offensive Rating Relative to League Average and better to have a lower **rDRTG_RS** or Defensive Rating Relative to League Average. Therefore, for the **rORTG_RS** scatterplot, the further away from the $y = x$ line the better and for the **rDRTG_RS** scatterplot, the further away from the $y = -x$ line the better.
* We will be applying a simple ordinary least squares regression line, so the higher the absolute value of the slope or the steeper the line the better.
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/59.embed" height="525" width="100%"></iframe> 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/61.embed" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/63.embed" height="525" width="100%"></iframe>

```python
# Merge our Matchup-Adjusted Playoff data with our League Average-Adjusted Regular Season data
df_merged = pd.merge(df_matchup_adj, df_rs_lg_adj,
                     how = 'left',
                     on = ['SEASON_YEAR', 'TEAM_ABBREVIATION'],
                     suffixes = ['_PO', '_RS'])
df_merged.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEASON_YEAR</th>
      <th>TEAM_ABBREVIATION</th>
      <th>SEASON_TYPE_PO</th>
      <th>WIN_PO</th>
      <th>WIN%_PO</th>
      <th>PACE_PO</th>
      <th>ORTG_PO</th>
      <th>DRTG_PO</th>
      <th>NRTG_PO</th>
      <th>OREB_RATE_PO</th>
      <th>DREB_RATE_PO</th>
      <th>DREB%_PO</th>
      <th>AST_RATE_PO</th>
      <th>%ASTD_PO</th>
      <th>STL_RATE_PO</th>
      <th>BLK_RATE_PO</th>
      <th>FGA_0_2_RATE_PO</th>
      <th>eFG_0_2%_PO</th>
      <th>FGA_NONRIM_2_RATE_PO</th>
      <th>eFG_NONRIM_2%_PO</th>
      <th>FGA_CORNER_3_RATE_PO</th>
      <th>eFG_CORNER_3%_PO</th>
      <th>FGA_24_26_RATE_PO</th>
      <th>eFG_24_26%_PO</th>
      <th>FGA_27_30_RATE_PO</th>
      <th>eFG_27_30%_PO</th>
      <th>FGA_31_RATE_PO</th>
      <th>eFG_31%_PO</th>
      <th>FG3A_RATE_PO</th>
      <th>FTA_RATE_PO</th>
      <th>TS%_PO</th>
      <th>eFG%_PO</th>
      <th>FG3%_PO</th>
      <th>FT%_PO</th>
      <th>TOV_RATE_PO</th>
      <th>LIVE_TOV_RATE_PO</th>
      <th>OPP_FGA_0_2_RATE_PO</th>
      <th>OPP_eFG_0_2%_PO</th>
      <th>OPP_FGA_NONRIM_2_RATE_PO</th>
      <th>OPP_eFG_NONRIM_2%_PO</th>
      <th>OPP_FGA_CORNER_3_RATE_PO</th>
      <th>OPP_eFG_CORNER_3%_PO</th>
      <th>OPP_FGA_24_26_RATE_PO</th>
      <th>OPP_eFG_24_26%_PO</th>
      <th>OPP_FGA_27_30_RATE_PO</th>
      <th>OPP_eFG_27_30%_PO</th>
      <th>OPP_FGA_31_RATE_PO</th>
      <th>OPP_eFG_31%_PO</th>
      <th>OPP_FG3A_RATE_PO</th>
      <th>OPP_FTA_RATE_PO</th>
      <th>OPP_TS%_PO</th>
      <th>OPP_eFG%_PO</th>
      <th>OPP_FG3%_PO</th>
      <th>OPP_FT%_PO</th>
      <th>OPP_TOV_RATE_PO</th>
      <th>OPP_LIVE_TOV_RATE_PO</th>
      <th>rORTG_PO</th>
      <th>rDRTG_PO</th>
      <th>rNRTG_PO</th>
      <th>TEAM_ID</th>
      <th>SEASON_TYPE_RS</th>
      <th>WIN_RS</th>
      <th>WIN%_RS</th>
      <th>PACE_RS</th>
      <th>ORTG_RS</th>
      <th>DRTG_RS</th>
      <th>NRTG_RS</th>
      <th>OREB_RATE_RS</th>
      <th>DREB_RATE_RS</th>
      <th>DREB%_RS</th>
      <th>AST_RATE_RS</th>
      <th>%ASTD_RS</th>
      <th>STL_RATE_RS</th>
      <th>BLK_RATE_RS</th>
      <th>FGA_0_2_RATE_RS</th>
      <th>eFG_0_2%_RS</th>
      <th>FGA_NONRIM_2_RATE_RS</th>
      <th>eFG_NONRIM_2%_RS</th>
      <th>FGA_CORNER_3_RATE_RS</th>
      <th>eFG_CORNER_3%_RS</th>
      <th>FGA_24_26_RATE_RS</th>
      <th>eFG_24_26%_RS</th>
      <th>FGA_27_30_RATE_RS</th>
      <th>eFG_27_30%_RS</th>
      <th>FGA_31_RATE_RS</th>
      <th>eFG_31%_RS</th>
      <th>FG3A_RATE_RS</th>
      <th>FTA_RATE_RS</th>
      <th>TS%_RS</th>
      <th>eFG%_RS</th>
      <th>FG3%_RS</th>
      <th>FT%_RS</th>
      <th>TOV_RATE_RS</th>
      <th>LIVE_TOV_RATE_RS</th>
      <th>OPP_FGA_0_2_RATE_RS</th>
      <th>OPP_eFG_0_2%_RS</th>
      <th>OPP_FGA_NONRIM_2_RATE_RS</th>
      <th>OPP_eFG_NONRIM_2%_RS</th>
      <th>OPP_FGA_CORNER_3_RATE_RS</th>
      <th>OPP_eFG_CORNER_3%_RS</th>
      <th>OPP_FGA_24_26_RATE_RS</th>
      <th>OPP_eFG_24_26%_RS</th>
      <th>OPP_FGA_27_30_RATE_RS</th>
      <th>OPP_eFG_27_30%_RS</th>
      <th>OPP_FGA_31_RATE_RS</th>
      <th>OPP_eFG_31%_RS</th>
      <th>OPP_FG3A_RATE_RS</th>
      <th>OPP_FTA_RATE_RS</th>
      <th>OPP_TS%_RS</th>
      <th>OPP_eFG%_RS</th>
      <th>OPP_FG3%_RS</th>
      <th>OPP_FT%_RS</th>
      <th>OPP_TOV_RATE_RS</th>
      <th>OPP_LIVE_TOV_RATE_RS</th>
      <th>WIN_LG_AVG</th>
      <th>WIN%_LG_AVG</th>
      <th>PACE_LG_AVG</th>
      <th>ORTG_LG_AVG</th>
      <th>DRTG_LG_AVG</th>
      <th>NRTG_LG_AVG</th>
      <th>OREB_RATE_LG_AVG</th>
      <th>DREB_RATE_LG_AVG</th>
      <th>DREB%_LG_AVG</th>
      <th>AST_RATE_LG_AVG</th>
      <th>%ASTD_LG_AVG</th>
      <th>STL_RATE_LG_AVG</th>
      <th>BLK_RATE_LG_AVG</th>
      <th>FGA_0_2_RATE_LG_AVG</th>
      <th>eFG_0_2%_LG_AVG</th>
      <th>FGA_NONRIM_2_RATE_LG_AVG</th>
      <th>eFG_NONRIM_2%_LG_AVG</th>
      <th>FGA_CORNER_3_RATE_LG_AVG</th>
      <th>eFG_CORNER_3%_LG_AVG</th>
      <th>FGA_24_26_RATE_LG_AVG</th>
      <th>eFG_24_26%_LG_AVG</th>
      <th>FGA_27_30_RATE_LG_AVG</th>
      <th>eFG_27_30%_LG_AVG</th>
      <th>FGA_31_RATE_LG_AVG</th>
      <th>eFG_31%_LG_AVG</th>
      <th>FG3A_RATE_LG_AVG</th>
      <th>FTA_RATE_LG_AVG</th>
      <th>TS%_LG_AVG</th>
      <th>eFG%_LG_AVG</th>
      <th>FG3%_LG_AVG</th>
      <th>FT%_LG_AVG</th>
      <th>TOV_RATE_LG_AVG</th>
      <th>LIVE_TOV_RATE_LG_AVG</th>
      <th>OPP_FGA_0_2_RATE_LG_AVG</th>
      <th>OPP_eFG_0_2%_LG_AVG</th>
      <th>OPP_FGA_NONRIM_2_RATE_LG_AVG</th>
      <th>OPP_eFG_NONRIM_2%_LG_AVG</th>
      <th>OPP_FGA_CORNER_3_RATE_LG_AVG</th>
      <th>OPP_eFG_CORNER_3%_LG_AVG</th>
      <th>OPP_FGA_24_26_RATE_LG_AVG</th>
      <th>OPP_eFG_24_26%_LG_AVG</th>
      <th>OPP_FGA_27_30_RATE_LG_AVG</th>
      <th>OPP_eFG_27_30%_LG_AVG</th>
      <th>OPP_FGA_31_RATE_LG_AVG</th>
      <th>OPP_eFG_31%_LG_AVG</th>
      <th>OPP_FG3A_RATE_LG_AVG</th>
      <th>OPP_FTA_RATE_LG_AVG</th>
      <th>OPP_TS%_LG_AVG</th>
      <th>OPP_eFG%_LG_AVG</th>
      <th>OPP_FG3%_LG_AVG</th>
      <th>OPP_FT%_LG_AVG</th>
      <th>OPP_TOV_RATE_LG_AVG</th>
      <th>OPP_LIVE_TOV_RATE_LG_AVG</th>
      <th>YEAR</th>
      <th>rWIN</th>
      <th>rWIN%</th>
      <th>rPACE</th>
      <th>rORTG_RS</th>
      <th>rDRTG_RS</th>
      <th>rNRTG_RS</th>
      <th>rOREB_RATE</th>
      <th>rDREB_RATE</th>
      <th>rDREB%</th>
      <th>rAST_RATE</th>
      <th>r%ASTD</th>
      <th>rSTL_RATE</th>
      <th>rBLK_RATE</th>
      <th>rFGA_0_2_RATE</th>
      <th>reFG_0_2%</th>
      <th>rFGA_NONRIM_2_RATE</th>
      <th>reFG_NONRIM_2%</th>
      <th>rFGA_CORNER_3_RATE</th>
      <th>reFG_CORNER_3%</th>
      <th>rFGA_24_26_RATE</th>
      <th>reFG_24_26%</th>
      <th>rFGA_27_30_RATE</th>
      <th>reFG_27_30%</th>
      <th>rFGA_31_RATE</th>
      <th>reFG_31%</th>
      <th>rFG3A_RATE</th>
      <th>rFTA_RATE</th>
      <th>rTS%</th>
      <th>reFG%</th>
      <th>rFG3%</th>
      <th>rFT%</th>
      <th>rTOV_RATE</th>
      <th>rLIVE_TOV_RATE</th>
      <th>rOPP_FGA_0_2_RATE</th>
      <th>rOPP_eFG_0_2%</th>
      <th>rOPP_FGA_NONRIM_2_RATE</th>
      <th>rOPP_eFG_NONRIM_2%</th>
      <th>rOPP_FGA_CORNER_3_RATE</th>
      <th>rOPP_eFG_CORNER_3%</th>
      <th>rOPP_FGA_24_26_RATE</th>
      <th>rOPP_eFG_24_26%</th>
      <th>rOPP_FGA_27_30_RATE</th>
      <th>rOPP_eFG_27_30%</th>
      <th>rOPP_FGA_31_RATE</th>
      <th>rOPP_eFG_31%</th>
      <th>rOPP_FG3A_RATE</th>
      <th>rOPP_FTA_RATE</th>
      <th>rOPP_TS%</th>
      <th>rOPP_eFG%</th>
      <th>rOPP_FG3%</th>
      <th>rOPP_FT%</th>
      <th>rOPP_TOV_RATE</th>
      <th>rOPP_LIVE_TOV_RATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-98</td>
      <td>ATL</td>
      <td>Playoffs</td>
      <td>1</td>
      <td>0.250000</td>
      <td>86.500000</td>
      <td>101.156069</td>
      <td>100.291545</td>
      <td>0.864524</td>
      <td>13.872832</td>
      <td>32.653061</td>
      <td>0.767123</td>
      <td>19.653179</td>
      <td>0.519084</td>
      <td>4.664723</td>
      <td>5.539359</td>
      <td>22.543353</td>
      <td>0.641026</td>
      <td>41.040462</td>
      <td>0.380282</td>
      <td>0.289017</td>
      <td>0.000000</td>
      <td>19.364162</td>
      <td>0.604478</td>
      <td>0.289017</td>
      <td>0.000000</td>
      <td>0.867052</td>
      <td>0.0</td>
      <td>20.809249</td>
      <td>25.722543</td>
      <td>0.522388</td>
      <td>0.494863</td>
      <td>0.375000</td>
      <td>0.685393</td>
      <td>14.161850</td>
      <td>9.826590</td>
      <td>23.906706</td>
      <td>0.646341</td>
      <td>48.688047</td>
      <td>0.425150</td>
      <td>1.166181</td>
      <td>1.125000</td>
      <td>8.163265</td>
      <td>0.589286</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.874636</td>
      <td>0.0</td>
      <td>10.204082</td>
      <td>25.947522</td>
      <td>0.532508</td>
      <td>0.510563</td>
      <td>0.400000</td>
      <td>0.606742</td>
      <td>14.577259</td>
      <td>10.787172</td>
      <td>-1.367715</td>
      <td>-4.941553</td>
      <td>3.573839</td>
      <td>1610612737</td>
      <td>Regular+Season</td>
      <td>50</td>
      <td>0.617284</td>
      <td>89.828921</td>
      <td>105.808734</td>
      <td>101.964261</td>
      <td>3.844472</td>
      <td>15.793770</td>
      <td>31.769199</td>
      <td>0.671375</td>
      <td>21.139981</td>
      <td>0.544118</td>
      <td>4.487792</td>
      <td>6.656663</td>
      <td>27.724119</td>
      <td>0.617272</td>
      <td>43.885186</td>
      <td>0.391507</td>
      <td>0.748198</td>
      <td>0.381818</td>
      <td>11.440620</td>
      <td>0.536861</td>
      <td>1.006666</td>
      <td>0.304054</td>
      <td>0.489729</td>
      <td>0.166667</td>
      <td>13.685213</td>
      <td>31.220242</td>
      <td>0.533580</td>
      <td>0.482060</td>
      <td>0.332008</td>
      <td>0.754684</td>
      <td>15.630526</td>
      <td>11.277377</td>
      <td>25.576320</td>
      <td>0.602133</td>
      <td>50.975310</td>
      <td>0.387209</td>
      <td>1.309508</td>
      <td>0.421875</td>
      <td>11.103533</td>
      <td>0.549140</td>
      <td>1.050334</td>
      <td>0.370130</td>
      <td>0.327377</td>
      <td>0.125000</td>
      <td>13.790752</td>
      <td>23.884872</td>
      <td>0.506642</td>
      <td>0.467240</td>
      <td>0.342235</td>
      <td>0.733866</td>
      <td>14.227254</td>
      <td>11.389988</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1129</td>
      <td>0.117284</td>
      <td>-2.413360</td>
      <td>2.966455</td>
      <td>-0.878017</td>
      <td>3.844472</td>
      <td>1.060290</td>
      <td>1.818031</td>
      <td>0.001096</td>
      <td>-2.568946</td>
      <td>-0.069576</td>
      <td>-0.035757</td>
      <td>1.203944</td>
      <td>1.477770</td>
      <td>-0.006783</td>
      <td>-1.897947</td>
      <td>0.008741</td>
      <td>-0.811678</td>
      <td>-0.172014</td>
      <td>0.748762</td>
      <td>0.001604</td>
      <td>-0.078128</td>
      <td>-0.116224</td>
      <td>0.152471</td>
      <td>0.053000</td>
      <td>0.011427</td>
      <td>2.924788</td>
      <td>0.010067</td>
      <td>0.003942</td>
      <td>-0.013229</td>
      <td>0.017631</td>
      <td>-0.339262</td>
      <td>-0.754386</td>
      <td>-0.670029</td>
      <td>-0.021921</td>
      <td>5.192178</td>
      <td>0.004444</td>
      <td>-0.250368</td>
      <td>-0.131958</td>
      <td>0.411675</td>
      <td>0.013883</td>
      <td>-0.034460</td>
      <td>-0.050148</td>
      <td>-0.009881</td>
      <td>0.011333</td>
      <td>0.116966</td>
      <td>-4.410582</td>
      <td>-0.016871</td>
      <td>-0.010878</td>
      <td>-0.003002</td>
      <td>-0.003186</td>
      <td>-1.742534</td>
      <td>-0.641776</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-98</td>
      <td>CHH</td>
      <td>Playoffs</td>
      <td>4</td>
      <td>0.444444</td>
      <td>84.777778</td>
      <td>97.640891</td>
      <td>103.363519</td>
      <td>-5.722628</td>
      <td>11.009174</td>
      <td>31.177232</td>
      <td>0.678873</td>
      <td>25.032765</td>
      <td>0.640940</td>
      <td>2.846054</td>
      <td>3.234153</td>
      <td>24.377457</td>
      <td>0.564516</td>
      <td>48.492792</td>
      <td>0.435135</td>
      <td>1.179554</td>
      <td>0.833333</td>
      <td>11.009174</td>
      <td>0.482143</td>
      <td>0.262123</td>
      <td>0.000000</td>
      <td>0.655308</td>
      <td>0.0</td>
      <td>13.106160</td>
      <td>22.411533</td>
      <td>0.509576</td>
      <td>0.478659</td>
      <td>0.320000</td>
      <td>0.684211</td>
      <td>15.203145</td>
      <td>11.664482</td>
      <td>25.614489</td>
      <td>0.621212</td>
      <td>44.501940</td>
      <td>0.389535</td>
      <td>0.905563</td>
      <td>0.642857</td>
      <td>14.489004</td>
      <td>0.549107</td>
      <td>1.164295</td>
      <td>0.166667</td>
      <td>0.776197</td>
      <td>0.0</td>
      <td>17.335058</td>
      <td>26.520052</td>
      <td>0.518831</td>
      <td>0.480030</td>
      <td>0.335821</td>
      <td>0.731707</td>
      <td>13.454075</td>
      <td>9.443726</td>
      <td>-2.191148</td>
      <td>-2.650596</td>
      <td>0.459448</td>
      <td>1610612766</td>
      <td>Regular+Season</td>
      <td>51</td>
      <td>0.621951</td>
      <td>91.352882</td>
      <td>105.233099</td>
      <td>102.523784</td>
      <td>2.709314</td>
      <td>13.653872</td>
      <td>30.364693</td>
      <td>0.692586</td>
      <td>25.780316</td>
      <td>0.653976</td>
      <td>4.862579</td>
      <td>4.122622</td>
      <td>27.453845</td>
      <td>0.638607</td>
      <td>44.800106</td>
      <td>0.386007</td>
      <td>1.779785</td>
      <td>0.492537</td>
      <td>9.563023</td>
      <td>0.600000</td>
      <td>0.411741</td>
      <td>0.532258</td>
      <td>0.252358</td>
      <td>0.236842</td>
      <td>12.006907</td>
      <td>29.034400</td>
      <td>0.542040</td>
      <td>0.495113</td>
      <td>0.382743</td>
      <td>0.750686</td>
      <td>15.858680</td>
      <td>11.528755</td>
      <td>25.079281</td>
      <td>0.658588</td>
      <td>45.626321</td>
      <td>0.393281</td>
      <td>1.532770</td>
      <td>0.543103</td>
      <td>11.257928</td>
      <td>0.536972</td>
      <td>0.898520</td>
      <td>0.397059</td>
      <td>0.356765</td>
      <td>0.222222</td>
      <td>14.045983</td>
      <td>25.951374</td>
      <td>0.533631</td>
      <td>0.492676</td>
      <td>0.347131</td>
      <td>0.731161</td>
      <td>15.776956</td>
      <td>11.720402</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1128</td>
      <td>0.121951</td>
      <td>-0.889399</td>
      <td>2.390820</td>
      <td>-0.318494</td>
      <td>2.709314</td>
      <td>-1.079608</td>
      <td>0.413525</td>
      <td>0.022307</td>
      <td>2.071389</td>
      <td>0.040282</td>
      <td>0.339031</td>
      <td>-1.330098</td>
      <td>1.207496</td>
      <td>0.014552</td>
      <td>-0.983026</td>
      <td>0.003241</td>
      <td>0.219909</td>
      <td>-0.061295</td>
      <td>-1.128835</td>
      <td>0.064743</td>
      <td>-0.673052</td>
      <td>0.111980</td>
      <td>-0.084901</td>
      <td>0.123175</td>
      <td>-1.666879</td>
      <td>0.738946</td>
      <td>0.018526</td>
      <td>0.016995</td>
      <td>0.037506</td>
      <td>0.013634</td>
      <td>-0.111108</td>
      <td>-0.503008</td>
      <td>-1.167068</td>
      <td>0.034533</td>
      <td>-0.156811</td>
      <td>0.010516</td>
      <td>-0.027106</td>
      <td>-0.010729</td>
      <td>0.566070</td>
      <td>0.001715</td>
      <td>-0.186274</td>
      <td>-0.023219</td>
      <td>0.019507</td>
      <td>0.108555</td>
      <td>0.372197</td>
      <td>-2.344080</td>
      <td>0.010118</td>
      <td>0.014558</td>
      <td>0.001893</td>
      <td>-0.005892</td>
      <td>-0.192833</td>
      <td>-0.311362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-98</td>
      <td>CHI</td>
      <td>Playoffs</td>
      <td>15</td>
      <td>0.714286</td>
      <td>86.286837</td>
      <td>106.885246</td>
      <td>100.611791</td>
      <td>6.273455</td>
      <td>16.994536</td>
      <td>30.533927</td>
      <td>0.709302</td>
      <td>23.551913</td>
      <td>0.601116</td>
      <td>6.729700</td>
      <td>5.283648</td>
      <td>26.830601</td>
      <td>0.613035</td>
      <td>46.338798</td>
      <td>0.386792</td>
      <td>1.038251</td>
      <td>0.631579</td>
      <td>12.349727</td>
      <td>0.511062</td>
      <td>0.928962</td>
      <td>0.264706</td>
      <td>0.546448</td>
      <td>0.0</td>
      <td>14.863388</td>
      <td>32.295082</td>
      <td>0.522436</td>
      <td>0.472377</td>
      <td>0.323529</td>
      <td>0.734349</td>
      <td>13.497268</td>
      <td>9.344262</td>
      <td>29.365962</td>
      <td>0.553030</td>
      <td>41.212458</td>
      <td>0.423752</td>
      <td>1.612903</td>
      <td>0.568966</td>
      <td>10.734149</td>
      <td>0.497409</td>
      <td>0.723026</td>
      <td>0.230769</td>
      <td>0.611791</td>
      <td>0.0</td>
      <td>13.681869</td>
      <td>27.141268</td>
      <td>0.521626</td>
      <td>0.476238</td>
      <td>0.313008</td>
      <td>0.750000</td>
      <td>16.629588</td>
      <td>12.625139</td>
      <td>5.189052</td>
      <td>-6.449337</td>
      <td>11.638389</td>
      <td>1610612741</td>
      <td>Regular+Season</td>
      <td>62</td>
      <td>0.765432</td>
      <td>90.388974</td>
      <td>106.180537</td>
      <td>98.090724</td>
      <td>8.089812</td>
      <td>17.362429</td>
      <td>31.658768</td>
      <td>0.684426</td>
      <td>26.172404</td>
      <td>0.638136</td>
      <td>4.644550</td>
      <td>4.725796</td>
      <td>28.097045</td>
      <td>0.618910</td>
      <td>49.945785</td>
      <td>0.390502</td>
      <td>1.545134</td>
      <td>0.513158</td>
      <td>9.677419</td>
      <td>0.518908</td>
      <td>1.165628</td>
      <td>0.313953</td>
      <td>0.311738</td>
      <td>0.000000</td>
      <td>12.699919</td>
      <td>26.877202</td>
      <td>0.517882</td>
      <td>0.474612</td>
      <td>0.324440</td>
      <td>0.745335</td>
      <td>14.638113</td>
      <td>11.574953</td>
      <td>27.433988</td>
      <td>0.600197</td>
      <td>45.280975</td>
      <td>0.360945</td>
      <td>1.178064</td>
      <td>0.448276</td>
      <td>10.656737</td>
      <td>0.516518</td>
      <td>1.435342</td>
      <td>0.396226</td>
      <td>0.473934</td>
      <td>0.085714</td>
      <td>13.744076</td>
      <td>26.174678</td>
      <td>0.500069</td>
      <td>0.456331</td>
      <td>0.322167</td>
      <td>0.730988</td>
      <td>16.303318</td>
      <td>12.227488</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1117</td>
      <td>0.265432</td>
      <td>-1.853307</td>
      <td>3.338259</td>
      <td>-4.751554</td>
      <td>8.089812</td>
      <td>2.628949</td>
      <td>1.707599</td>
      <td>0.014148</td>
      <td>2.463478</td>
      <td>0.024443</td>
      <td>0.121001</td>
      <td>-0.726924</td>
      <td>1.850696</td>
      <td>-0.005145</td>
      <td>4.162652</td>
      <td>0.007737</td>
      <td>-0.014742</td>
      <td>-0.040675</td>
      <td>-1.014439</td>
      <td>-0.016349</td>
      <td>0.080834</td>
      <td>-0.106324</td>
      <td>-0.025521</td>
      <td>-0.113667</td>
      <td>-0.973867</td>
      <td>-1.418252</td>
      <td>-0.005632</td>
      <td>-0.003506</td>
      <td>-0.020798</td>
      <td>0.008283</td>
      <td>-1.331675</td>
      <td>-0.456811</td>
      <td>1.187639</td>
      <td>-0.023857</td>
      <td>-0.502158</td>
      <td>-0.021820</td>
      <td>-0.381812</td>
      <td>-0.105557</td>
      <td>-0.035122</td>
      <td>-0.018739</td>
      <td>0.350548</td>
      <td>-0.024051</td>
      <td>0.136676</td>
      <td>-0.027953</td>
      <td>0.070290</td>
      <td>-2.120776</td>
      <td>-0.023445</td>
      <td>-0.021787</td>
      <td>-0.023070</td>
      <td>-0.006065</td>
      <td>0.333529</td>
      <td>0.195725</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-98</td>
      <td>CLE</td>
      <td>Playoffs</td>
      <td>1</td>
      <td>0.250000</td>
      <td>84.250000</td>
      <td>95.845697</td>
      <td>105.029586</td>
      <td>-9.183888</td>
      <td>12.166172</td>
      <td>28.698225</td>
      <td>0.723881</td>
      <td>21.958457</td>
      <td>0.691589</td>
      <td>7.100592</td>
      <td>3.550296</td>
      <td>22.255193</td>
      <td>0.640000</td>
      <td>43.620178</td>
      <td>0.353741</td>
      <td>0.890208</td>
      <td>0.500000</td>
      <td>5.637982</td>
      <td>0.473684</td>
      <td>0.296736</td>
      <td>0.000000</td>
      <td>0.296736</td>
      <td>0.0</td>
      <td>7.121662</td>
      <td>41.839763</td>
      <td>0.520968</td>
      <td>0.449187</td>
      <td>0.291667</td>
      <td>0.723404</td>
      <td>20.474777</td>
      <td>13.649852</td>
      <td>19.526627</td>
      <td>0.727273</td>
      <td>44.674556</td>
      <td>0.403974</td>
      <td>2.958580</td>
      <td>0.750000</td>
      <td>13.017751</td>
      <td>0.545455</td>
      <td>0.591716</td>
      <td>0.750000</td>
      <td>0.887574</td>
      <td>0.0</td>
      <td>17.455621</td>
      <td>26.923077</td>
      <td>0.567093</td>
      <td>0.514493</td>
      <td>0.372881</td>
      <td>0.780220</td>
      <td>16.568047</td>
      <td>11.834320</td>
      <td>-3.722279</td>
      <td>-1.103316</td>
      <td>-2.618963</td>
      <td>1610612739</td>
      <td>Regular+Season</td>
      <td>47</td>
      <td>0.573171</td>
      <td>91.715582</td>
      <td>100.092373</td>
      <td>96.931788</td>
      <td>3.160584</td>
      <td>13.103721</td>
      <td>30.194891</td>
      <td>0.700795</td>
      <td>24.993402</td>
      <td>0.672346</td>
      <td>5.109297</td>
      <td>5.517514</td>
      <td>25.098971</td>
      <td>0.620400</td>
      <td>46.225917</td>
      <td>0.382244</td>
      <td>1.662708</td>
      <td>0.714286</td>
      <td>7.878068</td>
      <td>0.547739</td>
      <td>0.699393</td>
      <td>0.537736</td>
      <td>0.329902</td>
      <td>0.060000</td>
      <td>10.570071</td>
      <td>28.859857</td>
      <td>0.528019</td>
      <td>0.477848</td>
      <td>0.372035</td>
      <td>0.755830</td>
      <td>17.973080</td>
      <td>12.140406</td>
      <td>23.505399</td>
      <td>0.629132</td>
      <td>45.193574</td>
      <td>0.356643</td>
      <td>1.158810</td>
      <td>0.562500</td>
      <td>10.824335</td>
      <td>0.534672</td>
      <td>0.829602</td>
      <td>0.333333</td>
      <td>0.250198</td>
      <td>0.078947</td>
      <td>13.062944</td>
      <td>28.641032</td>
      <td>0.512462</td>
      <td>0.460171</td>
      <td>0.343750</td>
      <td>0.754943</td>
      <td>18.290756</td>
      <td>13.932052</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1132</td>
      <td>0.073171</td>
      <td>-0.526698</td>
      <td>-2.749906</td>
      <td>-5.910490</td>
      <td>3.160584</td>
      <td>-1.629758</td>
      <td>0.243722</td>
      <td>0.030516</td>
      <td>1.284475</td>
      <td>0.058653</td>
      <td>0.585748</td>
      <td>0.064794</td>
      <td>-1.147378</td>
      <td>-0.003655</td>
      <td>0.442785</td>
      <td>-0.000521</td>
      <td>0.102832</td>
      <td>0.160453</td>
      <td>-2.813790</td>
      <td>0.012482</td>
      <td>-0.385401</td>
      <td>0.117458</td>
      <td>-0.007356</td>
      <td>-0.053667</td>
      <td>-3.103715</td>
      <td>0.564403</td>
      <td>0.004506</td>
      <td>-0.000270</td>
      <td>0.026798</td>
      <td>0.018777</td>
      <td>2.003292</td>
      <td>0.108643</td>
      <td>-2.740950</td>
      <td>0.005077</td>
      <td>-0.589559</td>
      <td>-0.026122</td>
      <td>-0.401066</td>
      <td>0.008667</td>
      <td>0.132477</td>
      <td>-0.000585</td>
      <td>-0.255191</td>
      <td>-0.086944</td>
      <td>-0.087061</td>
      <td>-0.034720</td>
      <td>-0.610842</td>
      <td>0.345578</td>
      <td>-0.011052</td>
      <td>-0.017947</td>
      <td>-0.001487</td>
      <td>0.017890</td>
      <td>2.320968</td>
      <td>1.900288</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-98</td>
      <td>HOU</td>
      <td>Playoffs</td>
      <td>2</td>
      <td>0.400000</td>
      <td>89.000000</td>
      <td>95.056180</td>
      <td>102.237136</td>
      <td>-7.180957</td>
      <td>13.483146</td>
      <td>34.004474</td>
      <td>0.737864</td>
      <td>18.202247</td>
      <td>0.558621</td>
      <td>3.131991</td>
      <td>5.592841</td>
      <td>19.775281</td>
      <td>0.545455</td>
      <td>44.044944</td>
      <td>0.372449</td>
      <td>1.797753</td>
      <td>0.187500</td>
      <td>18.202247</td>
      <td>0.425926</td>
      <td>0.224719</td>
      <td>0.000000</td>
      <td>0.674157</td>
      <td>0.0</td>
      <td>20.898876</td>
      <td>33.483146</td>
      <td>0.479592</td>
      <td>0.416446</td>
      <td>0.258065</td>
      <td>0.731544</td>
      <td>14.157303</td>
      <td>11.460674</td>
      <td>26.621924</td>
      <td>0.621849</td>
      <td>45.413870</td>
      <td>0.379310</td>
      <td>2.237136</td>
      <td>0.450000</td>
      <td>7.382550</td>
      <td>0.636364</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.447427</td>
      <td>0.0</td>
      <td>10.067114</td>
      <td>33.557047</td>
      <td>0.528935</td>
      <td>0.479620</td>
      <td>0.377778</td>
      <td>0.693333</td>
      <td>14.541387</td>
      <td>10.961969</td>
      <td>-7.349916</td>
      <td>-8.062136</td>
      <td>0.712220</td>
      <td>1610612745</td>
      <td>Regular+Season</td>
      <td>41</td>
      <td>0.500000</td>
      <td>92.792738</td>
      <td>105.634538</td>
      <td>106.138640</td>
      <td>-0.504102</td>
      <td>13.564628</td>
      <td>29.873846</td>
      <td>0.678181</td>
      <td>23.464197</td>
      <td>0.610659</td>
      <td>4.304851</td>
      <td>3.823644</td>
      <td>24.898917</td>
      <td>0.634887</td>
      <td>38.176601</td>
      <td>0.395285</td>
      <td>2.386853</td>
      <td>0.516393</td>
      <td>18.207904</td>
      <td>0.522206</td>
      <td>0.913004</td>
      <td>0.492857</td>
      <td>0.273901</td>
      <td>0.071429</td>
      <td>21.781662</td>
      <td>27.650972</td>
      <td>0.545204</td>
      <td>0.496315</td>
      <td>0.343114</td>
      <td>0.770755</td>
      <td>16.342768</td>
      <td>13.003782</td>
      <td>25.165821</td>
      <td>0.642377</td>
      <td>48.237742</td>
      <td>0.409814</td>
      <td>2.054884</td>
      <td>0.503165</td>
      <td>11.783067</td>
      <td>0.574503</td>
      <td>0.416179</td>
      <td>0.328125</td>
      <td>0.312134</td>
      <td>0.125000</td>
      <td>14.566263</td>
      <td>24.593575</td>
      <td>0.537615</td>
      <td>0.498966</td>
      <td>0.365179</td>
      <td>0.743522</td>
      <td>14.774353</td>
      <td>11.718039</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1138</td>
      <td>0.000000</td>
      <td>0.550458</td>
      <td>2.792259</td>
      <td>3.296361</td>
      <td>-0.504102</td>
      <td>-1.168852</td>
      <td>-0.077323</td>
      <td>0.007903</td>
      <td>-0.244729</td>
      <td>-0.003035</td>
      <td>-0.218698</td>
      <td>-1.629076</td>
      <td>-1.347432</td>
      <td>0.010833</td>
      <td>-7.606532</td>
      <td>0.012520</td>
      <td>0.826977</td>
      <td>-0.037439</td>
      <td>7.516046</td>
      <td>-0.013051</td>
      <td>-0.171790</td>
      <td>0.072579</td>
      <td>-0.063357</td>
      <td>-0.042239</td>
      <td>8.107876</td>
      <td>-0.644483</td>
      <td>0.021690</td>
      <td>0.018197</td>
      <td>-0.002124</td>
      <td>0.033702</td>
      <td>0.372979</td>
      <td>0.972019</td>
      <td>-1.080528</td>
      <td>0.018323</td>
      <td>2.454610</td>
      <td>0.027049</td>
      <td>0.495008</td>
      <td>-0.050668</td>
      <td>1.091208</td>
      <td>0.039246</td>
      <td>-0.668615</td>
      <td>-0.092153</td>
      <td>-0.025124</td>
      <td>0.011333</td>
      <td>0.892477</td>
      <td>-3.701879</td>
      <td>0.014102</td>
      <td>0.020848</td>
      <td>0.019941</td>
      <td>0.006469</td>
      <td>-1.195435</td>
      <td>-0.313725</td>
    </tr>
  </tbody>
</table>



#### A Non-Linear Relationship Between the Regular Season and Postseason
Notice, that despite there being clear trends, the $R^2$ values for each of these fits are well below 0.5. Furthermore, we do not really need data to tell us that good defensive teams generally defend well in the playoffs and good offensive teams generally score well in the playoffs.

Below, we look at correlation between teams' regular season and playoff features.


```python
cols_of_int = [col for col in df_teams.select_dtypes(include = [np.number]).columns.tolist() if col != 'WIN']
neut_cols = ['WIN%', 'NRTG', 'PACE']
def_cols = ['rDRTG'] + [col for col in cols_of_int if col[:4] == 'OPP_' or col in ['DRTG', 'DREB%', 'DREB_RATE', 'STL_RATE', 'BLK_RATE']]
off_cols = ['rORTG'] + [col for col in cols_of_int if col not in def_cols and col not in neut_cols]
```


```python
# Offensive features
off_corrs = []
for col in off_cols:
    off_corrs.append(df_merged[col + '_PO'].corr(df_merged[col + '_RS']))
off_corr_df = pd.DataFrame({'FEATURE': off_cols, 'CORR': off_corrs})
# Defensive features
def_corrs = []
for col in def_cols:
    def_corrs.append(df_merged[col + '_PO'].corr(df_merged[col + '_RS']))
def_corr_df = pd.DataFrame({'FEATURE': def_cols, 'CORR': def_corrs})
# Neutral features
neut_corrs = []
for col in neut_cols:
    neut_corrs.append(df_merged[col + '_PO'].corr(df_merged[col + '_RS']))
neut_corr_df = pd.DataFrame({'FEATURE': neut_cols, 'CORR': neut_corrs})
```

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/65.embed" height="525" width="100%"></iframe> 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/78.embed" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/80.embed" height="525" width="100%"></iframe>

First, we note most regular season features translate well to the postseason. Meaning that if a team is good at securing rebounds in the regular season there's a good chance they will also be good at it the postseason. However, note that that **three-point shooting** on both offense and defense have virtually not correlation between regular season and postseason.

This is because an offense's ability to make three-point shots and a defense's ability to defend three-point shots are extremely noisey. More intuitively, an uncontested or wide-open three-point shot is more likely to miss for reasons outside of the offense or defense's control than shots that are closer to the rim.


```python
# Offensive features
off_corrs = []
for col in off_cols:
    off_corrs.append(df_merged['ORTG_PO'].corr(df_merged[col + '_RS']))
off_corr_df = pd.DataFrame({'FEATURE': off_cols, 'CORR': off_corrs})
# Defensive features
def_corrs = []
for col in def_cols:
    def_corrs.append(df_merged['DRTG_PO'].corr(df_merged[col + '_RS']))
def_corr_df = pd.DataFrame({'FEATURE': def_cols, 'CORR': def_corrs})
```
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/82.embed" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/84.embed" height="525" width="100%"></iframe>

It is important to not place too much emphasis on correlation plots, which only display how features interact in a one-on-one setting. However, the lower correlations displayed above are a good representation of the challenge at hand. 

## 3. Modeling for Postseason Performance
We will be using random forests and gradient boosting, and hypertuning the parameters for those models.

For each methodology we will have three models to predict offensive, defensive, and net rating. The predictions from the offensive and defensive models will be combined to form a net rating and then compared to the net rating model.


```python
df_po = df_matchup_adj[['SEASON_YEAR', 'TEAM_ABBREVIATION', 'rORTG', 'rDRTG', 'rNRTG', 'ORTG', 'DRTG', 'NRTG']]
df_model = df_po.merge(df_rs_lg_adj, how = 'inner', on = ['SEASON_YEAR', 'TEAM_ABBREVIATION'], suffixes = ['_PO', '_RS'])
df_model.drop(['SEASON_TYPE', 'TEAM_ID', 'WIN'], axis = 1, inplace = True)
# cols = [col for col in df_model.columns if col[0] == 'r']
# df_model = df_model[cols].drop('rWIN', axis = 1)
df_model.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEASON_YEAR</th>
      <th>TEAM_ABBREVIATION</th>
      <th>rORTG_PO</th>
      <th>rDRTG_PO</th>
      <th>rNRTG_PO</th>
      <th>ORTG_PO</th>
      <th>DRTG_PO</th>
      <th>NRTG_PO</th>
      <th>WIN%</th>
      <th>PACE</th>
      <th>ORTG_RS</th>
      <th>DRTG_RS</th>
      <th>NRTG_RS</th>
      <th>OREB_RATE</th>
      <th>DREB_RATE</th>
      <th>DREB%</th>
      <th>AST_RATE</th>
      <th>%ASTD</th>
      <th>STL_RATE</th>
      <th>BLK_RATE</th>
      <th>FGA_0_2_RATE</th>
      <th>eFG_0_2%</th>
      <th>FGA_NONRIM_2_RATE</th>
      <th>eFG_NONRIM_2%</th>
      <th>FGA_CORNER_3_RATE</th>
      <th>eFG_CORNER_3%</th>
      <th>FGA_24_26_RATE</th>
      <th>eFG_24_26%</th>
      <th>FGA_27_30_RATE</th>
      <th>eFG_27_30%</th>
      <th>FGA_31_RATE</th>
      <th>eFG_31%</th>
      <th>FG3A_RATE</th>
      <th>FTA_RATE</th>
      <th>TS%</th>
      <th>eFG%</th>
      <th>FG3%</th>
      <th>FT%</th>
      <th>TOV_RATE</th>
      <th>LIVE_TOV_RATE</th>
      <th>OPP_FGA_0_2_RATE</th>
      <th>OPP_eFG_0_2%</th>
      <th>OPP_FGA_NONRIM_2_RATE</th>
      <th>OPP_eFG_NONRIM_2%</th>
      <th>OPP_FGA_CORNER_3_RATE</th>
      <th>OPP_eFG_CORNER_3%</th>
      <th>OPP_FGA_24_26_RATE</th>
      <th>OPP_eFG_24_26%</th>
      <th>OPP_FGA_27_30_RATE</th>
      <th>OPP_eFG_27_30%</th>
      <th>OPP_FGA_31_RATE</th>
      <th>OPP_eFG_31%</th>
      <th>OPP_FG3A_RATE</th>
      <th>OPP_FTA_RATE</th>
      <th>OPP_TS%</th>
      <th>OPP_eFG%</th>
      <th>OPP_FG3%</th>
      <th>OPP_FT%</th>
      <th>OPP_TOV_RATE</th>
      <th>OPP_LIVE_TOV_RATE</th>
      <th>WIN_LG_AVG</th>
      <th>WIN%_LG_AVG</th>
      <th>PACE_LG_AVG</th>
      <th>ORTG_LG_AVG</th>
      <th>DRTG_LG_AVG</th>
      <th>NRTG_LG_AVG</th>
      <th>OREB_RATE_LG_AVG</th>
      <th>DREB_RATE_LG_AVG</th>
      <th>DREB%_LG_AVG</th>
      <th>AST_RATE_LG_AVG</th>
      <th>%ASTD_LG_AVG</th>
      <th>STL_RATE_LG_AVG</th>
      <th>BLK_RATE_LG_AVG</th>
      <th>FGA_0_2_RATE_LG_AVG</th>
      <th>eFG_0_2%_LG_AVG</th>
      <th>FGA_NONRIM_2_RATE_LG_AVG</th>
      <th>eFG_NONRIM_2%_LG_AVG</th>
      <th>FGA_CORNER_3_RATE_LG_AVG</th>
      <th>eFG_CORNER_3%_LG_AVG</th>
      <th>FGA_24_26_RATE_LG_AVG</th>
      <th>eFG_24_26%_LG_AVG</th>
      <th>FGA_27_30_RATE_LG_AVG</th>
      <th>eFG_27_30%_LG_AVG</th>
      <th>FGA_31_RATE_LG_AVG</th>
      <th>eFG_31%_LG_AVG</th>
      <th>FG3A_RATE_LG_AVG</th>
      <th>FTA_RATE_LG_AVG</th>
      <th>TS%_LG_AVG</th>
      <th>eFG%_LG_AVG</th>
      <th>FG3%_LG_AVG</th>
      <th>FT%_LG_AVG</th>
      <th>TOV_RATE_LG_AVG</th>
      <th>LIVE_TOV_RATE_LG_AVG</th>
      <th>OPP_FGA_0_2_RATE_LG_AVG</th>
      <th>OPP_eFG_0_2%_LG_AVG</th>
      <th>OPP_FGA_NONRIM_2_RATE_LG_AVG</th>
      <th>OPP_eFG_NONRIM_2%_LG_AVG</th>
      <th>OPP_FGA_CORNER_3_RATE_LG_AVG</th>
      <th>OPP_eFG_CORNER_3%_LG_AVG</th>
      <th>OPP_FGA_24_26_RATE_LG_AVG</th>
      <th>OPP_eFG_24_26%_LG_AVG</th>
      <th>OPP_FGA_27_30_RATE_LG_AVG</th>
      <th>OPP_eFG_27_30%_LG_AVG</th>
      <th>OPP_FGA_31_RATE_LG_AVG</th>
      <th>OPP_eFG_31%_LG_AVG</th>
      <th>OPP_FG3A_RATE_LG_AVG</th>
      <th>OPP_FTA_RATE_LG_AVG</th>
      <th>OPP_TS%_LG_AVG</th>
      <th>OPP_eFG%_LG_AVG</th>
      <th>OPP_FG3%_LG_AVG</th>
      <th>OPP_FT%_LG_AVG</th>
      <th>OPP_TOV_RATE_LG_AVG</th>
      <th>OPP_LIVE_TOV_RATE_LG_AVG</th>
      <th>YEAR</th>
      <th>rWIN</th>
      <th>rWIN%</th>
      <th>rPACE</th>
      <th>rORTG_RS</th>
      <th>rDRTG_RS</th>
      <th>rNRTG_RS</th>
      <th>rOREB_RATE</th>
      <th>rDREB_RATE</th>
      <th>rDREB%</th>
      <th>rAST_RATE</th>
      <th>r%ASTD</th>
      <th>rSTL_RATE</th>
      <th>rBLK_RATE</th>
      <th>rFGA_0_2_RATE</th>
      <th>reFG_0_2%</th>
      <th>rFGA_NONRIM_2_RATE</th>
      <th>reFG_NONRIM_2%</th>
      <th>rFGA_CORNER_3_RATE</th>
      <th>reFG_CORNER_3%</th>
      <th>rFGA_24_26_RATE</th>
      <th>reFG_24_26%</th>
      <th>rFGA_27_30_RATE</th>
      <th>reFG_27_30%</th>
      <th>rFGA_31_RATE</th>
      <th>reFG_31%</th>
      <th>rFG3A_RATE</th>
      <th>rFTA_RATE</th>
      <th>rTS%</th>
      <th>reFG%</th>
      <th>rFG3%</th>
      <th>rFT%</th>
      <th>rTOV_RATE</th>
      <th>rLIVE_TOV_RATE</th>
      <th>rOPP_FGA_0_2_RATE</th>
      <th>rOPP_eFG_0_2%</th>
      <th>rOPP_FGA_NONRIM_2_RATE</th>
      <th>rOPP_eFG_NONRIM_2%</th>
      <th>rOPP_FGA_CORNER_3_RATE</th>
      <th>rOPP_eFG_CORNER_3%</th>
      <th>rOPP_FGA_24_26_RATE</th>
      <th>rOPP_eFG_24_26%</th>
      <th>rOPP_FGA_27_30_RATE</th>
      <th>rOPP_eFG_27_30%</th>
      <th>rOPP_FGA_31_RATE</th>
      <th>rOPP_eFG_31%</th>
      <th>rOPP_FG3A_RATE</th>
      <th>rOPP_FTA_RATE</th>
      <th>rOPP_TS%</th>
      <th>rOPP_eFG%</th>
      <th>rOPP_FG3%</th>
      <th>rOPP_FT%</th>
      <th>rOPP_TOV_RATE</th>
      <th>rOPP_LIVE_TOV_RATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-98</td>
      <td>ATL</td>
      <td>-1.367715</td>
      <td>-4.941553</td>
      <td>3.573839</td>
      <td>101.156069</td>
      <td>100.291545</td>
      <td>0.864524</td>
      <td>0.617284</td>
      <td>89.828921</td>
      <td>105.808734</td>
      <td>101.964261</td>
      <td>3.844472</td>
      <td>15.793770</td>
      <td>31.769199</td>
      <td>0.671375</td>
      <td>21.139981</td>
      <td>0.544118</td>
      <td>4.487792</td>
      <td>6.656663</td>
      <td>27.724119</td>
      <td>0.617272</td>
      <td>43.885186</td>
      <td>0.391507</td>
      <td>0.748198</td>
      <td>0.381818</td>
      <td>11.440620</td>
      <td>0.536861</td>
      <td>1.006666</td>
      <td>0.304054</td>
      <td>0.489729</td>
      <td>0.166667</td>
      <td>13.685213</td>
      <td>31.220242</td>
      <td>0.533580</td>
      <td>0.482060</td>
      <td>0.332008</td>
      <td>0.754684</td>
      <td>15.630526</td>
      <td>11.277377</td>
      <td>25.576320</td>
      <td>0.602133</td>
      <td>50.975310</td>
      <td>0.387209</td>
      <td>1.309508</td>
      <td>0.421875</td>
      <td>11.103533</td>
      <td>0.549140</td>
      <td>1.050334</td>
      <td>0.370130</td>
      <td>0.327377</td>
      <td>0.125000</td>
      <td>13.790752</td>
      <td>23.884872</td>
      <td>0.506642</td>
      <td>0.467240</td>
      <td>0.342235</td>
      <td>0.733866</td>
      <td>14.227254</td>
      <td>11.389988</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1129</td>
      <td>0.117284</td>
      <td>-2.413360</td>
      <td>2.966455</td>
      <td>-0.878017</td>
      <td>3.844472</td>
      <td>1.060290</td>
      <td>1.818031</td>
      <td>0.001096</td>
      <td>-2.568946</td>
      <td>-0.069576</td>
      <td>-0.035757</td>
      <td>1.203944</td>
      <td>1.477770</td>
      <td>-0.006783</td>
      <td>-1.897947</td>
      <td>0.008741</td>
      <td>-0.811678</td>
      <td>-0.172014</td>
      <td>0.748762</td>
      <td>0.001604</td>
      <td>-0.078128</td>
      <td>-0.116224</td>
      <td>0.152471</td>
      <td>0.053000</td>
      <td>0.011427</td>
      <td>2.924788</td>
      <td>0.010067</td>
      <td>0.003942</td>
      <td>-0.013229</td>
      <td>0.017631</td>
      <td>-0.339262</td>
      <td>-0.754386</td>
      <td>-0.670029</td>
      <td>-0.021921</td>
      <td>5.192178</td>
      <td>0.004444</td>
      <td>-0.250368</td>
      <td>-0.131958</td>
      <td>0.411675</td>
      <td>0.013883</td>
      <td>-0.034460</td>
      <td>-0.050148</td>
      <td>-0.009881</td>
      <td>0.011333</td>
      <td>0.116966</td>
      <td>-4.410582</td>
      <td>-0.016871</td>
      <td>-0.010878</td>
      <td>-0.003002</td>
      <td>-0.003186</td>
      <td>-1.742534</td>
      <td>-0.641776</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-98</td>
      <td>CHH</td>
      <td>-2.191148</td>
      <td>-2.650596</td>
      <td>0.459448</td>
      <td>97.640891</td>
      <td>103.363519</td>
      <td>-5.722628</td>
      <td>0.621951</td>
      <td>91.352882</td>
      <td>105.233099</td>
      <td>102.523784</td>
      <td>2.709314</td>
      <td>13.653872</td>
      <td>30.364693</td>
      <td>0.692586</td>
      <td>25.780316</td>
      <td>0.653976</td>
      <td>4.862579</td>
      <td>4.122622</td>
      <td>27.453845</td>
      <td>0.638607</td>
      <td>44.800106</td>
      <td>0.386007</td>
      <td>1.779785</td>
      <td>0.492537</td>
      <td>9.563023</td>
      <td>0.600000</td>
      <td>0.411741</td>
      <td>0.532258</td>
      <td>0.252358</td>
      <td>0.236842</td>
      <td>12.006907</td>
      <td>29.034400</td>
      <td>0.542040</td>
      <td>0.495113</td>
      <td>0.382743</td>
      <td>0.750686</td>
      <td>15.858680</td>
      <td>11.528755</td>
      <td>25.079281</td>
      <td>0.658588</td>
      <td>45.626321</td>
      <td>0.393281</td>
      <td>1.532770</td>
      <td>0.543103</td>
      <td>11.257928</td>
      <td>0.536972</td>
      <td>0.898520</td>
      <td>0.397059</td>
      <td>0.356765</td>
      <td>0.222222</td>
      <td>14.045983</td>
      <td>25.951374</td>
      <td>0.533631</td>
      <td>0.492676</td>
      <td>0.347131</td>
      <td>0.731161</td>
      <td>15.776956</td>
      <td>11.720402</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1128</td>
      <td>0.121951</td>
      <td>-0.889399</td>
      <td>2.390820</td>
      <td>-0.318494</td>
      <td>2.709314</td>
      <td>-1.079608</td>
      <td>0.413525</td>
      <td>0.022307</td>
      <td>2.071389</td>
      <td>0.040282</td>
      <td>0.339031</td>
      <td>-1.330098</td>
      <td>1.207496</td>
      <td>0.014552</td>
      <td>-0.983026</td>
      <td>0.003241</td>
      <td>0.219909</td>
      <td>-0.061295</td>
      <td>-1.128835</td>
      <td>0.064743</td>
      <td>-0.673052</td>
      <td>0.111980</td>
      <td>-0.084901</td>
      <td>0.123175</td>
      <td>-1.666879</td>
      <td>0.738946</td>
      <td>0.018526</td>
      <td>0.016995</td>
      <td>0.037506</td>
      <td>0.013634</td>
      <td>-0.111108</td>
      <td>-0.503008</td>
      <td>-1.167068</td>
      <td>0.034533</td>
      <td>-0.156811</td>
      <td>0.010516</td>
      <td>-0.027106</td>
      <td>-0.010729</td>
      <td>0.566070</td>
      <td>0.001715</td>
      <td>-0.186274</td>
      <td>-0.023219</td>
      <td>0.019507</td>
      <td>0.108555</td>
      <td>0.372197</td>
      <td>-2.344080</td>
      <td>0.010118</td>
      <td>0.014558</td>
      <td>0.001893</td>
      <td>-0.005892</td>
      <td>-0.192833</td>
      <td>-0.311362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-98</td>
      <td>CHI</td>
      <td>5.189052</td>
      <td>-6.449337</td>
      <td>11.638389</td>
      <td>106.885246</td>
      <td>100.611791</td>
      <td>6.273455</td>
      <td>0.765432</td>
      <td>90.388974</td>
      <td>106.180537</td>
      <td>98.090724</td>
      <td>8.089812</td>
      <td>17.362429</td>
      <td>31.658768</td>
      <td>0.684426</td>
      <td>26.172404</td>
      <td>0.638136</td>
      <td>4.644550</td>
      <td>4.725796</td>
      <td>28.097045</td>
      <td>0.618910</td>
      <td>49.945785</td>
      <td>0.390502</td>
      <td>1.545134</td>
      <td>0.513158</td>
      <td>9.677419</td>
      <td>0.518908</td>
      <td>1.165628</td>
      <td>0.313953</td>
      <td>0.311738</td>
      <td>0.000000</td>
      <td>12.699919</td>
      <td>26.877202</td>
      <td>0.517882</td>
      <td>0.474612</td>
      <td>0.324440</td>
      <td>0.745335</td>
      <td>14.638113</td>
      <td>11.574953</td>
      <td>27.433988</td>
      <td>0.600197</td>
      <td>45.280975</td>
      <td>0.360945</td>
      <td>1.178064</td>
      <td>0.448276</td>
      <td>10.656737</td>
      <td>0.516518</td>
      <td>1.435342</td>
      <td>0.396226</td>
      <td>0.473934</td>
      <td>0.085714</td>
      <td>13.744076</td>
      <td>26.174678</td>
      <td>0.500069</td>
      <td>0.456331</td>
      <td>0.322167</td>
      <td>0.730988</td>
      <td>16.303318</td>
      <td>12.227488</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1117</td>
      <td>0.265432</td>
      <td>-1.853307</td>
      <td>3.338259</td>
      <td>-4.751554</td>
      <td>8.089812</td>
      <td>2.628949</td>
      <td>1.707599</td>
      <td>0.014148</td>
      <td>2.463478</td>
      <td>0.024443</td>
      <td>0.121001</td>
      <td>-0.726924</td>
      <td>1.850696</td>
      <td>-0.005145</td>
      <td>4.162652</td>
      <td>0.007737</td>
      <td>-0.014742</td>
      <td>-0.040675</td>
      <td>-1.014439</td>
      <td>-0.016349</td>
      <td>0.080834</td>
      <td>-0.106324</td>
      <td>-0.025521</td>
      <td>-0.113667</td>
      <td>-0.973867</td>
      <td>-1.418252</td>
      <td>-0.005632</td>
      <td>-0.003506</td>
      <td>-0.020798</td>
      <td>0.008283</td>
      <td>-1.331675</td>
      <td>-0.456811</td>
      <td>1.187639</td>
      <td>-0.023857</td>
      <td>-0.502158</td>
      <td>-0.021820</td>
      <td>-0.381812</td>
      <td>-0.105557</td>
      <td>-0.035122</td>
      <td>-0.018739</td>
      <td>0.350548</td>
      <td>-0.024051</td>
      <td>0.136676</td>
      <td>-0.027953</td>
      <td>0.070290</td>
      <td>-2.120776</td>
      <td>-0.023445</td>
      <td>-0.021787</td>
      <td>-0.023070</td>
      <td>-0.006065</td>
      <td>0.333529</td>
      <td>0.195725</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-98</td>
      <td>CLE</td>
      <td>-3.722279</td>
      <td>-1.103316</td>
      <td>-2.618963</td>
      <td>95.845697</td>
      <td>105.029586</td>
      <td>-9.183888</td>
      <td>0.573171</td>
      <td>91.715582</td>
      <td>100.092373</td>
      <td>96.931788</td>
      <td>3.160584</td>
      <td>13.103721</td>
      <td>30.194891</td>
      <td>0.700795</td>
      <td>24.993402</td>
      <td>0.672346</td>
      <td>5.109297</td>
      <td>5.517514</td>
      <td>25.098971</td>
      <td>0.620400</td>
      <td>46.225917</td>
      <td>0.382244</td>
      <td>1.662708</td>
      <td>0.714286</td>
      <td>7.878068</td>
      <td>0.547739</td>
      <td>0.699393</td>
      <td>0.537736</td>
      <td>0.329902</td>
      <td>0.060000</td>
      <td>10.570071</td>
      <td>28.859857</td>
      <td>0.528019</td>
      <td>0.477848</td>
      <td>0.372035</td>
      <td>0.755830</td>
      <td>17.973080</td>
      <td>12.140406</td>
      <td>23.505399</td>
      <td>0.629132</td>
      <td>45.193574</td>
      <td>0.356643</td>
      <td>1.158810</td>
      <td>0.562500</td>
      <td>10.824335</td>
      <td>0.534672</td>
      <td>0.829602</td>
      <td>0.333333</td>
      <td>0.250198</td>
      <td>0.078947</td>
      <td>13.062944</td>
      <td>28.641032</td>
      <td>0.512462</td>
      <td>0.460171</td>
      <td>0.343750</td>
      <td>0.754943</td>
      <td>18.290756</td>
      <td>13.932052</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1132</td>
      <td>0.073171</td>
      <td>-0.526698</td>
      <td>-2.749906</td>
      <td>-5.910490</td>
      <td>3.160584</td>
      <td>-1.629758</td>
      <td>0.243722</td>
      <td>0.030516</td>
      <td>1.284475</td>
      <td>0.058653</td>
      <td>0.585748</td>
      <td>0.064794</td>
      <td>-1.147378</td>
      <td>-0.003655</td>
      <td>0.442785</td>
      <td>-0.000521</td>
      <td>0.102832</td>
      <td>0.160453</td>
      <td>-2.813790</td>
      <td>0.012482</td>
      <td>-0.385401</td>
      <td>0.117458</td>
      <td>-0.007356</td>
      <td>-0.053667</td>
      <td>-3.103715</td>
      <td>0.564403</td>
      <td>0.004506</td>
      <td>-0.000270</td>
      <td>0.026798</td>
      <td>0.018777</td>
      <td>2.003292</td>
      <td>0.108643</td>
      <td>-2.740950</td>
      <td>0.005077</td>
      <td>-0.589559</td>
      <td>-0.026122</td>
      <td>-0.401066</td>
      <td>0.008667</td>
      <td>0.132477</td>
      <td>-0.000585</td>
      <td>-0.255191</td>
      <td>-0.086944</td>
      <td>-0.087061</td>
      <td>-0.034720</td>
      <td>-0.610842</td>
      <td>0.345578</td>
      <td>-0.011052</td>
      <td>-0.017947</td>
      <td>-0.001487</td>
      <td>0.017890</td>
      <td>2.320968</td>
      <td>1.900288</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-98</td>
      <td>HOU</td>
      <td>-7.349916</td>
      <td>-8.062136</td>
      <td>0.712220</td>
      <td>95.056180</td>
      <td>102.237136</td>
      <td>-7.180957</td>
      <td>0.500000</td>
      <td>92.792738</td>
      <td>105.634538</td>
      <td>106.138640</td>
      <td>-0.504102</td>
      <td>13.564628</td>
      <td>29.873846</td>
      <td>0.678181</td>
      <td>23.464197</td>
      <td>0.610659</td>
      <td>4.304851</td>
      <td>3.823644</td>
      <td>24.898917</td>
      <td>0.634887</td>
      <td>38.176601</td>
      <td>0.395285</td>
      <td>2.386853</td>
      <td>0.516393</td>
      <td>18.207904</td>
      <td>0.522206</td>
      <td>0.913004</td>
      <td>0.492857</td>
      <td>0.273901</td>
      <td>0.071429</td>
      <td>21.781662</td>
      <td>27.650972</td>
      <td>0.545204</td>
      <td>0.496315</td>
      <td>0.343114</td>
      <td>0.770755</td>
      <td>16.342768</td>
      <td>13.003782</td>
      <td>25.165821</td>
      <td>0.642377</td>
      <td>48.237742</td>
      <td>0.409814</td>
      <td>2.054884</td>
      <td>0.503165</td>
      <td>11.783067</td>
      <td>0.574503</td>
      <td>0.416179</td>
      <td>0.328125</td>
      <td>0.312134</td>
      <td>0.125000</td>
      <td>14.566263</td>
      <td>24.593575</td>
      <td>0.537615</td>
      <td>0.498966</td>
      <td>0.365179</td>
      <td>0.743522</td>
      <td>14.774353</td>
      <td>11.718039</td>
      <td>1179</td>
      <td>0.5</td>
      <td>92.242281</td>
      <td>102.842278</td>
      <td>102.842278</td>
      <td>0.0</td>
      <td>14.733479</td>
      <td>29.951168</td>
      <td>0.670279</td>
      <td>23.708927</td>
      <td>0.613694</td>
      <td>4.523549</td>
      <td>5.45272</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>26.246349</td>
      <td>0.624055</td>
      <td>45.783133</td>
      <td>0.382765</td>
      <td>1.559876</td>
      <td>0.553833</td>
      <td>10.691858</td>
      <td>0.535257</td>
      <td>1.084794</td>
      <td>0.420278</td>
      <td>0.337258</td>
      <td>0.113667</td>
      <td>13.673786</td>
      <td>28.295455</td>
      <td>0.523514</td>
      <td>0.478118</td>
      <td>0.345237</td>
      <td>0.737053</td>
      <td>15.969788</td>
      <td>12.031763</td>
      <td>1997</td>
      <td>-1138</td>
      <td>0.000000</td>
      <td>0.550458</td>
      <td>2.792259</td>
      <td>3.296361</td>
      <td>-0.504102</td>
      <td>-1.168852</td>
      <td>-0.077323</td>
      <td>0.007903</td>
      <td>-0.244729</td>
      <td>-0.003035</td>
      <td>-0.218698</td>
      <td>-1.629076</td>
      <td>-1.347432</td>
      <td>0.010833</td>
      <td>-7.606532</td>
      <td>0.012520</td>
      <td>0.826977</td>
      <td>-0.037439</td>
      <td>7.516046</td>
      <td>-0.013051</td>
      <td>-0.171790</td>
      <td>0.072579</td>
      <td>-0.063357</td>
      <td>-0.042239</td>
      <td>8.107876</td>
      <td>-0.644483</td>
      <td>0.021690</td>
      <td>0.018197</td>
      <td>-0.002124</td>
      <td>0.033702</td>
      <td>0.372979</td>
      <td>0.972019</td>
      <td>-1.080528</td>
      <td>0.018323</td>
      <td>2.454610</td>
      <td>0.027049</td>
      <td>0.495008</td>
      <td>-0.050668</td>
      <td>1.091208</td>
      <td>0.039246</td>
      <td>-0.668615</td>
      <td>-0.092153</td>
      <td>-0.025124</td>
      <td>0.011333</td>
      <td>0.892477</td>
      <td>-3.701879</td>
      <td>0.014102</td>
      <td>0.020848</td>
      <td>0.019941</td>
      <td>0.006469</td>
      <td>-1.195435</td>
      <td>-0.313725</td>
    </tr>
  </tbody>
</table>




```python
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
```

#### Prepping the data and defining functions for modeling
We first split our data into a training and test set and then, we split our features into offense and defense. Features will be normalized to their respective season's league average and then scaled. We will scale our training set separately from the validation and test set to prevent data leakage.


```python
stdSc = StandardScaler()
```


```python
y = df_model[['rNRTG_PO', 'rORTG_PO', 'rDRTG_PO']]
x = df_model.drop(['rNRTG_RS', 'rORTG_PO', 'rDRTG_PO', 'SEASON_YEAR', 'TEAM_ABBREVIATION'], axis = 1)
```


```python
# Standard Boxscore
base_off = ['WIN%', 'ORTG_RS', 'rORTG_RS', 'OREB_RATE', 'AST_RATE', '%ASTD', 'FG3A_RATE', 'FTA_RATE', 
            'TS%', 'FG3%', 'LIVE_TOV_RATE']
            #'FGA_0_2_RATE', 'eFG_0_2%', 'FGA_CORNER_3_RATE']
base_def = ['WIN%', 'DRTG_RS', 'rDRTG_RS', 'DREB%', 'BLK_RATE', 'OPP_LIVE_TOV_RATE',
            'OPP_FG3A_RATE', 'OPP_FTA_RATE', 'OPP_TS%']
            #'OPP_FGA_0_2_RATE', 'OPP_eFG_0_2%', 'OPP_FGA_CORNER_3_RATE']
base = list(set(base_off + base_def))
# Relative League Average
rel_off = ['WIN%', 'ORTG_RS', 'rORTG_RS'] + ['r' + col for col in base_off if col not in ['WIN%', 'ORTG_RS', 'rORTG_RS']]
rel_def = ['WIN%', 'DRTG_RS', 'rDRTG_RS'] + ['r' + col for col in base_def if col not in ['WIN%', 'DRTG_RS', 'rDRTG_RS', 'rDRTG_FT_ADJ', 'rDRTG_FG3_ADJ', 'rDRTG_FG3_FT_ADJ']]
rel = list(set(rel_off + rel_def))

resps = ['rNRTG_PO', 'rORTG_PO', 'rDRTG_PO']
feats = [rel, rel_off, rel_def]
```


```python
x = df_model[rel]
y = df_model[resps]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 10)
```


```python
training_data, test_data = {}, {}
for feat, resp in zip(feats, resps):
    # Scale
    x_tr = stdSc.fit_transform(x_train[feat])
    x_te = stdSc.transform(x_test[feat])
    
    training_data[resp] = [x_tr, y_train[resp]]
    test_data[resp] = [x_te, y_test[resp]]
```

Below we define a few functions to help with the parameter tuning.


```python
# Fit on training data and predict for test data without hyperparameter tuning
def modeler(model, params, data, resp, scale = True):
    mod = model(**params)
    mod.fit(training_data[resp][0], training_data[resp][1])
    train_rmse = np.sqrt(-cross_val_score(mod, training_data[resp][0], training_data[resp][1], 
                          scoring = 'neg_mean_squared_error', cv = 10))
    
    pred = mod.predict(test_data[resp][0])
    test_rmse = (sum((pred - test_data[resp][1])**2)/len(pred))**0.5
    return (train_rmse, test_rmse, pred)

# Hyperparameter tuner using validation set
def grid_searcher(model, model_params, param_search, 
                  resp, scorer = 'neg_root_mean_squared_error', CV = 10):
    grid_search = GridSearchCV(estimator = model(**model_params),
                              param_grid = param_search,
                              scoring = scorer,
                              n_jobs = -1,
                              cv = CV)
    grid_search.fit(training_data[resp][0], training_data[resp][1])
    return grid_search
```

## Gradient Boosted Regression Model

Below we will define our baseline model using sklearn's default parameters in order to give us a basis of comparison once the parameter tuning is done.


```python
results = []
columns = ['MODEL', 'RESPONSE', 'TRAIN_RMSE', 'TRAIN_RMSE_STD', 'TEST_RMSE']
predictions = []
```


```python
gb_params = {'random_state': 10}
for feat, resp in zip(feats, resps):
    gb = modeler(GradientBoostingRegressor, gb_params, df_model, resp, feat)
    print(r'For {resp}, train RMSE is {train:.{dec}f} and test RMSE is {test:.{dec}f}'.format(dec = 2, resp = resp, train = gb[0].mean(), test = gb[1]))
    results += [['GB - Base', resp, gb[0].mean(), gb[0].std(), gb[1]]]
    predictions += [['GB - Base', resp, i] for i in gb[2]]
```

    For rNRTG_PO, train RMSE is 5.86 and test RMSE is 7.10
    For rORTG_PO, train RMSE is 4.86 and test RMSE is 4.60
    For rDRTG_PO, train RMSE is 4.72 and test RMSE is 5.22
    

### Parameter Tuning for Gradient Boosting
Parameters for gradient boosting can be separated into tree-specific parameters -- we are primarily interested in max_depth, num_samples_split, min_samples_leaf, max_features --, boosting-specific parameters -- learning rate, n_estimators, and subsample --, and miscellaneous parameters.

We will tune the tree-specific parameters by holding the boosting-specific parameters constant and then tune the boosting-specific parameters.
#### Setting fixed values for n_estimators and learning_rate


```python
gb_params = {'learning_rate': 0.05,
             'min_samples_split': 5, 'min_samples_leaf': 5, 
             'max_depth': 5, 'max_features': 'sqrt', 
             'subsample': 0.8, 'random_state': 10}
param_search = {'n_estimators': range(20, 101, 10)}
grids = []
for feat, resp in zip(feats, resps):
    grids.append(grid_searcher(GradientBoostingRegressor, gb_params, param_search, resp))
```


```python
n_ests = []
for i in range(len(grids)):
    print(resps[i])
    print(grids[i].best_params_)
    #print(pd.DataFrame({'n_estimators': [param.values() for param in grids[i].cv_results_['params']], 
    #                  'mean_test_score': grids[i].cv_results_['mean_test_score'], 
    #                  'std_test_score': grids[i].cv_results_['std_test_score']}))
    n_ests.append(grids[i].best_params_['n_estimators'])
```

    rNRTG_PO
    {'n_estimators': 30}
    rORTG_PO
    {'n_estimators': 40}
    rDRTG_PO
    {'n_estimators': 30}
    

### Tuning Tree-Specific Parameters
We will be tuning the parameters in the following order: (1) max_depth and min_samples_split, (2) min_samples_leaf, and then (3) max_features. We do so to tune the most important parameters first.
#### Tuning max_depth and min_samples_split 


```python
grids = []
for feat, resp, n_est in zip(feats, resps, n_ests):
    gb_params = {'learning_rate': 0.05, 'n_estimators': n_est,
                 'min_samples_leaf': 5, 'max_features': 'sqrt', 
                 'subsample': 0.8, 'random_state': 10}
    param_search = {'min_samples_split': range(5, 101, 5), 'max_depth': range(2, 13, 2)}
    grids.append(grid_searcher(GradientBoostingRegressor, gb_params, param_search, resp))

min_samples_splits = []
max_depths = []
for i in range(len(grids)):
    print(resps[i])
    #print(pd.DataFrame({'params': [param.values() for param in grids[i].cv_results_['params']], 
    #                  'mean_test_score': grids[i].cv_results_['mean_test_score'], 
    #                  'std_test_score': grids[i].cv_results_['std_test_score']}))
    print(grids[i].best_params_)
    min_samples_splits.append(grids[i].best_params_['min_samples_split'])
    max_depths.append(grids[i].best_params_['max_depth'])
```

    rNRTG_PO
    {'max_depth': 4, 'min_samples_split': 95}
    rORTG_PO
    {'max_depth': 4, 'min_samples_split': 35}
    rDRTG_PO
    {'max_depth': 4, 'min_samples_split': 70}
    

#### Tuning min_samples_leaf


```python
grids = []
for feat, resp, n_est, min_samples_split, max_depth in zip(feats, resps, n_ests, min_samples_splits, max_depths):
    gb_params = {'learning_rate': 0.05, 'n_estimators': n_est, 'max_depth': max_depth,
                 'min_samples_split': min_samples_split, 'max_features': 'sqrt', 
                 'subsample': 0.8, 'random_state': 10}
    param_search = {'min_samples_leaf': range(5, 101, 5)}
    grids.append(grid_searcher(GradientBoostingRegressor, gb_params, param_search, resp))

min_samples_leaves = []
for i in range(len(grids)):
    print(resps[i])
    #print(pd.DataFrame({'params': [param.values() for param in grids[i].cv_results_['params']], 
    #                  'mean_test_score': grids[i].cv_results_['mean_test_score'], 
    #                  'std_test_score': grids[i].cv_results_['std_test_score']}))
    print(grids[i].best_params_)
    min_samples_leaves.append(grids[i].best_params_['min_samples_leaf'])
```

    rNRTG_PO
    {'min_samples_leaf': 5}
    rORTG_PO
    {'min_samples_leaf': 5}
    rDRTG_PO
    {'min_samples_leaf': 5}
    

#### Tuning max_features


```python
grids = []
for feat, resp, n_est, min_samples_split, max_depth, min_samples_leaf in zip(feats, resps, n_ests, min_samples_splits, max_depths, min_samples_leaves):
    gb_params = {'learning_rate': 0.05, 'n_estimators': n_est, 'max_depth': max_depth,
                 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 
                 'subsample': 0.8, 'random_state': 10}
    param_search = {'max_features': range(3,len(feat))}
    grids.append(grid_searcher(GradientBoostingRegressor, gb_params, param_search, resp))

max_features = []
for i in range(len(grids)):
    print(resps[i])
    #print(pd.DataFrame({'params': [param.values() for param in grids[i].cv_results_['params']], 
    #                  'mean_test_score': grids[i].cv_results_['mean_test_score'], 
    #                  'std_test_score': grids[i].cv_results_['std_test_score']}))
    print(grids[i].best_params_)
    max_features.append(grids[i].best_params_['max_features'])
```

    rNRTG_PO
    {'max_features': 14}
    rORTG_PO
    {'max_features': 3}
    rDRTG_PO
    {'max_features': 6}
    

#### Tuning subsample


```python
tree_params = []
for n_est, min_samples_split, max_depth, min_samples_leaf, max_feature in zip(n_ests, min_samples_splits, max_depths, min_samples_leaves, max_features):
    tree_params.append({'learning_rate': 0.05,
                        'n_estimators': n_est, 'min_samples_split': min_samples_split,
                        'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
                        'max_features': max_feature, 'random_state': 10})
```


```python
grids = []
for resp, tree_param in zip(resps, tree_params):
    param_search = {'subsample': np.linspace(0.5, .95, 10)}
    grids.append(grid_searcher(GradientBoostingRegressor, tree_param, param_search, resp))
    subsamples = []
for i in range(len(grids)):
    print(resps[i])
    #print(pd.DataFrame({'params': [param.values() for param in grids[i].cv_results_['params']], 
    #                  'mean_test_score': grids[i].cv_results_['mean_test_score'], 
    #                  'std_test_score': grids[i].cv_results_['std_test_score']}))
    print(grids[i].best_params_)
    subsamples.append(grids[i].best_params_['subsample'])
```

    rNRTG_PO
    {'subsample': 0.8}
    rORTG_PO
    {'subsample': 0.6}
    rDRTG_PO
    {'subsample': 0.75}
    

#### Tuning n_estimators and learning_rate
We will tune the learning rate and number of trees by lowering learning rate by a factor and raising the number of trees by the reciprocal of that factor. So, if we halve the learning rate we will double the number of trees used by the gradient boosted model.


```python
tree_params = []
for min_samples_split, max_depth, min_samples_leaf, max_feature, subsample in zip(min_samples_splits, max_depths, min_samples_leaves, max_features, subsamples):
    tree_params.append({'min_samples_split': min_samples_split,
                        'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
                        'max_features': max_feature, 'subsample': subsample, 'random_state': 10})
```

Response, training RMSE, standard deviation of training RMSE, and test RMSE are displayed below for different combinations of learning rate and number of trees.


```python
factor = 1
tuning_results = []
for feat, resp, tree_param, n_est in zip(feats, resps, tree_params, n_ests):
    tree_param['learning_rate'] = 0.05 * factor
    tree_param['n_estimators'] = int(n_est/factor)
    gb = modeler(GradientBoostingRegressor, tree_param, df_model, resp, feat)
    tuning_results.append([resp, gb[0].mean(), gb[0].std(), gb[1]])
tuning_results
```




    [['rNRTG_PO', 5.311379871164113, 0.6304517326365363, 6.029804749822923],
     ['rORTG_PO', 4.448248672154153, 0.7158030033977415, 4.308650657311786],
     ['rDRTG_PO', 4.392670737843608, 0.6992141432551986, 4.880966342687494]]




```python
factor = 0.5
tuning_results = []
for feat, resp, tree_param, n_est in zip(feats, resps, tree_params, n_ests):
    tree_param['learning_rate'] = 0.05 * factor
    tree_param['n_estimators'] = int(n_est/factor)
    gb = modeler(GradientBoostingRegressor, tree_param, df_model, resp, feat)
    tuning_results.append([resp, gb[0].mean(), gb[0].std(), gb[1]])
tuning_results
```




    [['rNRTG_PO', 5.352986752216157, 0.6413322522382954, 6.016499192487262],
     ['rORTG_PO', 4.506475937746671, 0.7327778981025368, 4.2787925377714515],
     ['rDRTG_PO', 4.415084273285731, 0.7033819831463909, 4.889981508100448]]




```python
factor = 0.25
tuning_results = []
for feat, resp, tree_param, n_est in zip(feats, resps, tree_params, n_ests):
    tree_param['learning_rate'] = 0.05 * factor
    tree_param['n_estimators'] = int(n_est/factor)
    gb = modeler(GradientBoostingRegressor, tree_param, df_model, resp, feat)
    tuning_results.append([resp, gb[0].mean(), gb[0].std(), gb[1]])
tuning_results
```




    [['rNRTG_PO', 5.345612646049235, 0.6368055723311822, 6.030292999517835],
     ['rORTG_PO', 4.486913828373719, 0.7242338340275892, 4.271524824289904],
     ['rDRTG_PO', 4.417227372082785, 0.6871949159244694, 4.899709862559187]]




```python
tree_params = []
for n_est, min_samples_split, max_depth, min_samples_leaf, max_feature, subsample in zip(n_ests, min_samples_splits, max_depths, min_samples_leaves, max_features, subsamples):
    tree_params.append({'n_estimators': n_est, 'learning_rate': 0.05,
                        'min_samples_split': min_samples_split,
                        'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
                        'max_features': max_feature, 'subsample': subsample, 'random_state': 10})
```


```python
for feat, resp, tree_param in zip(feats, resps, tree_params):
    gb = modeler(GradientBoostingRegressor, tree_param, df_model, resp, feat)
    print(r'For {resp}, train RMSE is {train:.{dec}f} and test RMSE is {test:.{dec}f}'.format(dec = 2, resp = resp, train = gb[0].mean(), test = gb[1]))
    results += [['GB - Param Tune', resp, gb[0].mean(), gb[0].std(), gb[1]]]
    predictions += [['GB - Param Tune', resp, i] for i in gb[2]]
```

    For rNRTG_PO, train RMSE is 5.31 and test RMSE is 6.03
    For rORTG_PO, train RMSE is 4.45 and test RMSE is 4.31
    For rDRTG_PO, train RMSE is 4.39 and test RMSE is 4.88
    


```python
gb_pred_df = pd.DataFrame(predictions, columns = ['MODEL', 'RESP', 'PRED'])
```
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/74.embed" height="525" width="100%"></iframe>

### Results of Hyperparameter Tuning for the Gradient Boosted Model
* The gradient boosted model's performance improves noticeably with parameter tuning, however, the predictive power of these models are not great.
* The gradient boosted model overfits for net rating and defensive rating, and has a very small discrepancy between training and test RMSE for offensive rating.
* It is very likely that hyperparameter tuning has lead to overfitting. Ideally, we would proceed with a nested cross-validation procedure for hyperparameter tuning to reduce the bias seen in the results above. However, given the overall poor performance the computational cost does not seem worth effort.




```python
lr_feats = ['rNRTG_RS', 'rORTG_RS', 'rDRTG_RS']
lr_params = {}
for lr_feat, resp in zip(lr_feats, resps):
    lr = modeler(LinearRegression, lr_params, df_model, resp, lr_feat)
    results.append(['Linear Regression', resp, lr[0].mean(), lr[0].std(), lr[1]])
    predictions += [['Linear Regression', resp, i] for i in lr[2]]
```


```python
gb_df.MODEL.unique()
```




    array(['Gradient Boosting - Base', 'Gradient Boosting - Parameter Tuning',
           'RF - Base', 'Linear Regression'], dtype=object)

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/76.embed" height="525" width="100%"></iframe>

A gradient boosted model with hyperparameter more or less performs the same as a simple linear regression model. Even when accounting for potential overfitting due to the hyperparameter tuning, this is a very clear indication that something has gone wrong.

## 4. Understanding What Went Wrong
Initially, this was meant to solely be an exploration on methods for properly evaluating regular season and postseason performance. It was only while sifting through the results that I decided on attempting to model for postseason performance.

It is likely that better feature selection and engineering, and more careful parameter tuning and model selection (a few other models were attempted, but gradient boosting provided the best results) would have led to better performance, but I am not sure *how* much better those results would have been.
### Accounting for Matchups
In evaluating how well teams do in the playoffs, we compared their performance to their opponents' regular season metrics. In other words, we accounted for the quality of competition. Our model, however, had no way to account for the actual opponents teams faced.

A possible solution to this is build a model that outputs the probability of a team winning a playoff series against specific opponents and then run simulations of the playoffs.
### Going Down a Level of Detail
In the NBA, a single player can have an outsized effect on the outcome of a game, moreso than all other team-based professional sports. This is in part because each team only has five players on the court. The above model relies on team-level data and therefore, doesn't account for individual players. 

There are a couple of easy fixes that may prove worthwhile, however I believe the most effective approach is to establish a metric for true player value. For those unfamiliar, assessing player value, particularly on defense, is a very involved and complex task.

For those interested, the following are explainers for two player impact metrics -- PIPM (Player Impact Plus-Minus) and RAPM (Regularized-Adjusted Plus-Minus).
* https://www.bball-index.com/player-impact-plus-minus/
* https://squared2020.com/2017/09/18/deep-dive-on-regularized-adjusted-plus-minus-i-introductory-example/

