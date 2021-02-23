---
title: "Handling Early-Season Trends in the NBA"
layout: single
classes: wide

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

In the following notebook, I will be exploring some early(-ish) season trends of the 2020-21 season (data includes games played **up to February 11th, 2021**) and methods for evaluating and projecting how teams and players will perform for the *rest* of the season.

The contents of the notebook are as follows:
1. **The Current State of the 2020-21 NBA Season**: evaluating the NBA standings as of February 11th and how much each team has either improved or decline from the previous season
2. **Reducing Noise in Defensive Rating**: two noisey aspects of defense, opponent free-throw and three-point percentage, are replaced with their respective league averages to attain a truer representation of defensive ability
3. **Mean Regression for Binary Variables**: mean regression is applied to binary variables to stabilise early season trends
4. **Building Credible Intervals using Empirical Bayes Estimation**: the beta-binomial conjugate-prior is used to handle player shooting percentages throughout the season
5. **Conclusion**: possible ways to improve the methods used in the notebook are gone over

It should be noted that the majority of the code used in the original notebook is omitted and the repository for the notebook can be accessed [here](https://github.com/achen719/Early-Season-Trends).

Data has been scraped and wrangled from the NBA's official stats database. It contains boxscores (game-level summaries) for each team and player from the 1997-98 season to the current 2020-21 season. The data has also been further cleaned and manipulated to fit the purposes of this notebook. The dataset of player-level boxscores has been 

Snippets of team-level and player-level boxscores are supplied below:
#### Team-level boxscores
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
      <th>OREB</th>
      <th>DREB</th>
      <th>TOV</th>
      <th>BAD_PASS_TOV</th>
      <th>LIVE_TOV</th>
      <th>FG_0_2</th>
      <th>FG_3_9</th>
      <th>FG_10_15</th>
      <th>FG_16_3PT</th>
      <th>FG_CORNER_3</th>
      <th>FG_24_26</th>
      <th>FG_27_30</th>
      <th>FG_31_35</th>
      <th>FG_36</th>
      <th>FGA_0_2</th>
      <th>FGA_3_9</th>
      <th>FGA_10_15</th>
      <th>FGA_16_3PT</th>
      <th>FGA_CORNER_3</th>
      <th>FGA_24_26</th>
      <th>FGA_27_30</th>
      <th>FGA_31_35</th>
      <th>FGA_36</th>
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
      <th>OPP_OREB</th>
      <th>OPP_DREB</th>
      <th>OPP_TOV</th>
      <th>OPP_BAD_PASS_TOV</th>
      <th>OPP_LIVE_TOV</th>
      <th>OPP_OFF_FOUL</th>
      <th>OPP_FG_0_2</th>
      <th>OPP_FG_3_9</th>
      <th>OPP_FG_10_15</th>
      <th>OPP_FG_16_3PT</th>
      <th>OPP_FG_CORNER_3</th>
      <th>OPP_FG_24_26</th>
      <th>OPP_FG_27_30</th>
      <th>OPP_FG_31_35</th>
      <th>OPP_FG_36</th>
      <th>OPP_FGA_0_2</th>
      <th>OPP_FGA_3_9</th>
      <th>OPP_FGA_10_15</th>
      <th>OPP_FGA_16_3PT</th>
      <th>OPP_FGA_CORNER_3</th>
      <th>OPP_FGA_24_26</th>
      <th>OPP_FGA_27_30</th>
      <th>OPP_FGA_31_35</th>
      <th>OPP_FGA_36</th>
      <th>OPP_AST</th>
      <th>OPP_STL</th>
      <th>OPP_BLK</th>
      <th>GAMES</th>
      <th>WIN</th>
      <th>OPP_FG2</th>
      <th>OPP_FG2A</th>
      <th>OPPONENT</th>
      <th>FG2</th>
      <th>FG2A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29700004</td>
      <td>1997-10-31</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>AWAY</td>
      <td>ATL - ORL</td>
      <td>28800</td>
      <td>105</td>
      <td>72</td>
      <td>40</td>
      <td>27</td>
      <td>23</td>
      <td>7</td>
      <td>2</td>
      <td>13</td>
      <td>30</td>
      <td>16</td>
      <td>10</td>
      <td>12</td>
      <td>20</td>
      <td>4</td>
      <td>5</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>31</td>
      <td>8</td>
      <td>11</td>
      <td>15</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>0</td>
      <td>5</td>
      <td>94</td>
      <td>93</td>
      <td>99</td>
      <td>78</td>
      <td>37</td>
      <td>30</td>
      <td>24</td>
      <td>9</td>
      <td>1</td>
      <td>10</td>
      <td>19</td>
      <td>10</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>20</td>
      <td>6</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>11</td>
      <td>9</td>
      <td>19</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>36</td>
      <td>69</td>
      <td>ORL</td>
      <td>38</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29700016</td>
      <td>1997-11-01</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>HOME</td>
      <td>ATL - TOR</td>
      <td>28800</td>
      <td>90</td>
      <td>72</td>
      <td>35</td>
      <td>27</td>
      <td>20</td>
      <td>6</td>
      <td>0</td>
      <td>14</td>
      <td>36</td>
      <td>21</td>
      <td>10</td>
      <td>15</td>
      <td>26</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>14</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>19</td>
      <td>4</td>
      <td>10</td>
      <td>94</td>
      <td>92</td>
      <td>85</td>
      <td>92</td>
      <td>35</td>
      <td>15</td>
      <td>11</td>
      <td>14</td>
      <td>4</td>
      <td>16</td>
      <td>19</td>
      <td>10</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>18</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>12</td>
      <td>17</td>
      <td>22</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>31</td>
      <td>78</td>
      <td>TOR</td>
      <td>35</td>
      <td>66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29700033</td>
      <td>1997-11-04</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>HOME</td>
      <td>ATL - DET</td>
      <td>28800</td>
      <td>82</td>
      <td>69</td>
      <td>31</td>
      <td>22</td>
      <td>16</td>
      <td>10</td>
      <td>4</td>
      <td>12</td>
      <td>39</td>
      <td>19</td>
      <td>8</td>
      <td>14</td>
      <td>14</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>8</td>
      <td>18</td>
      <td>13</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>4</td>
      <td>5</td>
      <td>81</td>
      <td>84</td>
      <td>71</td>
      <td>75</td>
      <td>28</td>
      <td>19</td>
      <td>14</td>
      <td>6</td>
      <td>1</td>
      <td>9</td>
      <td>21</td>
      <td>8</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>9</td>
      <td>8</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>13</td>
      <td>18</td>
      <td>20</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>27</td>
      <td>69</td>
      <td>DET</td>
      <td>27</td>
      <td>59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29700045</td>
      <td>1997-11-05</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>AWAY</td>
      <td>ATL - PHI</td>
      <td>28800</td>
      <td>93</td>
      <td>77</td>
      <td>36</td>
      <td>19</td>
      <td>15</td>
      <td>13</td>
      <td>6</td>
      <td>17</td>
      <td>29</td>
      <td>19</td>
      <td>10</td>
      <td>14</td>
      <td>14</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>16</td>
      <td>20</td>
      <td>8</td>
      <td>0</td>
      <td>8</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>12</td>
      <td>89</td>
      <td>90</td>
      <td>88</td>
      <td>77</td>
      <td>35</td>
      <td>23</td>
      <td>15</td>
      <td>14</td>
      <td>3</td>
      <td>13</td>
      <td>21</td>
      <td>16</td>
      <td>5</td>
      <td>13</td>
      <td>0</td>
      <td>14</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>17</td>
      <td>16</td>
      <td>11</td>
      <td>0</td>
      <td>6</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>32</td>
      <td>63</td>
      <td>PHI</td>
      <td>30</td>
      <td>64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29700057</td>
      <td>1997-11-07</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>HOME</td>
      <td>ATL - CHI</td>
      <td>28800</td>
      <td>80</td>
      <td>79</td>
      <td>29</td>
      <td>21</td>
      <td>19</td>
      <td>11</td>
      <td>3</td>
      <td>15</td>
      <td>32</td>
      <td>14</td>
      <td>5</td>
      <td>11</td>
      <td>10</td>
      <td>3</td>
      <td>8</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>11</td>
      <td>24</td>
      <td>17</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>9</td>
      <td>8</td>
      <td>88</td>
      <td>89</td>
      <td>78</td>
      <td>80</td>
      <td>32</td>
      <td>16</td>
      <td>12</td>
      <td>8</td>
      <td>2</td>
      <td>7</td>
      <td>29</td>
      <td>12</td>
      <td>4</td>
      <td>10</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>14</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>5</td>
      <td>32</td>
      <td>24</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>30</td>
      <td>72</td>
      <td>CHI</td>
      <td>26</td>
      <td>68</td>
    </tr>
  </tbody>
</table>

#### Player-level boxscores

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GAME_ID</th>
      <th>GAME_DATE</th>
      <th>SEASON_YEAR</th>
      <th>SEASON_TYPE</th>
      <th>PLAYER_ID</th>
      <th>PLAYER_NAME</th>
      <th>TEAM_ID</th>
      <th>TEAM_ABBREVIATION</th>
      <th>MATCHUP</th>
      <th>PTS</th>
      <th>FGA</th>
      <th>FG</th>
      <th>FTA</th>
      <th>FT</th>
      <th>FG3A</th>
      <th>FG3</th>
      <th>ON_PTS</th>
      <th>ON_OPP_PTS</th>
      <th>ON_OFF_POSS</th>
      <th>ON_DEF_POSS</th>
      <th>GAMES</th>
      <th>OPPONENT</th>
      <th>FG2</th>
      <th>FG2A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29700587</td>
      <td>1998-01-24</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1003</td>
      <td>Drew Barry</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>ATL - POR</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>14</td>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>POR</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29700608</td>
      <td>1998-01-27</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1003</td>
      <td>Drew Barry</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>ATL - MIN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>MIN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29700639</td>
      <td>1998-01-31</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1003</td>
      <td>Drew Barry</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>ATL - CHH</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>CHH</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29700679</td>
      <td>1998-02-05</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1003</td>
      <td>Drew Barry</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>ATL - CLE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>CLE</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29700712</td>
      <td>1998-02-13</td>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1003</td>
      <td>Drew Barry</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>ATL - CHI</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>29</td>
      <td>15</td>
      <td>22</td>
      <td>19</td>
      <td>1</td>
      <td>CHI</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

This data is also aggregated to the season-level for each player and team with additional calculated fields. Below are additional snippets of the data:

#### Team-level aggregation for each season

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEASON_YEAR</th>
      <th>SEASON_TYPE</th>
      <th>TEAM_ID</th>
      <th>TEAM_ABBREVIATION</th>
      <th>MIN_PLAYED</th>
      <th>PTS</th>
      <th>FGA</th>
      <th>FG</th>
      <th>FTA</th>
      <th>FT</th>
      <th>FG3A</th>
      <th>FG3</th>
      <th>OREB</th>
      <th>DREB</th>
      <th>TOV</th>
      <th>BAD_PASS_TOV</th>
      <th>LIVE_TOV</th>
      <th>FG_0_2</th>
      <th>FG_3_9</th>
      <th>FG_10_15</th>
      <th>FG_16_3PT</th>
      <th>FG_CORNER_3</th>
      <th>FG_24_26</th>
      <th>FG_27_30</th>
      <th>FG_31_35</th>
      <th>FG_36</th>
      <th>FGA_0_2</th>
      <th>FGA_3_9</th>
      <th>FGA_10_15</th>
      <th>FGA_16_3PT</th>
      <th>FGA_CORNER_3</th>
      <th>FGA_24_26</th>
      <th>FGA_27_30</th>
      <th>FGA_31_35</th>
      <th>FGA_36</th>
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
      <th>OPP_OREB</th>
      <th>OPP_DREB</th>
      <th>OPP_TOV</th>
      <th>OPP_BAD_PASS_TOV</th>
      <th>OPP_LIVE_TOV</th>
      <th>OPP_OFF_FOUL</th>
      <th>OPP_FG_0_2</th>
      <th>OPP_FG_3_9</th>
      <th>OPP_FG_10_15</th>
      <th>OPP_FG_16_3PT</th>
      <th>OPP_FG_CORNER_3</th>
      <th>OPP_FG_24_26</th>
      <th>OPP_FG_27_30</th>
      <th>OPP_FG_31_35</th>
      <th>OPP_FG_36</th>
      <th>OPP_FGA_0_2</th>
      <th>OPP_FGA_3_9</th>
      <th>OPP_FGA_10_15</th>
      <th>OPP_FGA_16_3PT</th>
      <th>OPP_FGA_CORNER_3</th>
      <th>OPP_FGA_24_26</th>
      <th>OPP_FGA_27_30</th>
      <th>OPP_FGA_31_35</th>
      <th>OPP_FGA_36</th>
      <th>OPP_AST</th>
      <th>OPP_STL</th>
      <th>OPP_BLK</th>
      <th>GAMES</th>
      <th>WIN</th>
      <th>OPP_FG2</th>
      <th>OPP_FG2A</th>
      <th>FG2</th>
      <th>FG2A</th>
      <th>WIN%</th>
      <th>ORTG</th>
      <th>DRTG</th>
      <th>NRTG</th>
      <th>FG2A_RATE</th>
      <th>FG3A_RATE</th>
      <th>FTA_RATE</th>
      <th>FG%</th>
      <th>FG2%</th>
      <th>FG3%</th>
      <th>FT%</th>
      <th>OPP_FG2A_RATE</th>
      <th>OPP_FG3A_RATE</th>
      <th>OPP_FTA_RATE</th>
      <th>OPP_FG%</th>
      <th>OPP_FG2%</th>
      <th>OPP_FG3%</th>
      <th>OPP_FT%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>2356800</td>
      <td>7778</td>
      <td>6271</td>
      <td>2856</td>
      <td>2295</td>
      <td>1732</td>
      <td>1006</td>
      <td>334</td>
      <td>1161</td>
      <td>2329</td>
      <td>1149</td>
      <td>500</td>
      <td>829</td>
      <td>1258</td>
      <td>342</td>
      <td>400</td>
      <td>521</td>
      <td>14</td>
      <td>301</td>
      <td>15</td>
      <td>0</td>
      <td>4</td>
      <td>2038</td>
      <td>960</td>
      <td>1075</td>
      <td>1191</td>
      <td>55</td>
      <td>841</td>
      <td>74</td>
      <td>8</td>
      <td>28</td>
      <td>1554</td>
      <td>329</td>
      <td>488</td>
      <td>7351</td>
      <td>7331</td>
      <td>7475</td>
      <td>6624</td>
      <td>2922</td>
      <td>1751</td>
      <td>1285</td>
      <td>1011</td>
      <td>346</td>
      <td>1140</td>
      <td>2051</td>
      <td>1043</td>
      <td>442</td>
      <td>835</td>
      <td>0</td>
      <td>1129</td>
      <td>369</td>
      <td>534</td>
      <td>544</td>
      <td>27</td>
      <td>298</td>
      <td>19</td>
      <td>0</td>
      <td>2</td>
      <td>1875</td>
      <td>1060</td>
      <td>1363</td>
      <td>1314</td>
      <td>96</td>
      <td>814</td>
      <td>77</td>
      <td>6</td>
      <td>18</td>
      <td>1724</td>
      <td>293</td>
      <td>363</td>
      <td>81</td>
      <td>50</td>
      <td>2576</td>
      <td>5613</td>
      <td>2522</td>
      <td>5265</td>
      <td>0.617284</td>
      <td>105.808734</td>
      <td>101.964261</td>
      <td>3.844472</td>
      <td>71.622908</td>
      <td>13.685213</td>
      <td>31.220242</td>
      <td>0.455430</td>
      <td>0.479012</td>
      <td>0.332008</td>
      <td>0.754684</td>
      <td>76.565271</td>
      <td>13.790752</td>
      <td>23.884872</td>
      <td>0.441123</td>
      <td>0.458935</td>
      <td>0.342235</td>
      <td>0.733866</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612738</td>
      <td>BOS</td>
      <td>2367600</td>
      <td>7864</td>
      <td>6934</td>
      <td>3012</td>
      <td>1964</td>
      <td>1425</td>
      <td>1249</td>
      <td>415</td>
      <td>1220</td>
      <td>2020</td>
      <td>1276</td>
      <td>574</td>
      <td>958</td>
      <td>1227</td>
      <td>421</td>
      <td>294</td>
      <td>655</td>
      <td>26</td>
      <td>363</td>
      <td>23</td>
      <td>3</td>
      <td>0</td>
      <td>2046</td>
      <td>1204</td>
      <td>859</td>
      <td>1573</td>
      <td>80</td>
      <td>1046</td>
      <td>97</td>
      <td>8</td>
      <td>18</td>
      <td>1816</td>
      <td>508</td>
      <td>366</td>
      <td>7894</td>
      <td>7829</td>
      <td>8079</td>
      <td>5981</td>
      <td>2864</td>
      <td>2756</td>
      <td>2052</td>
      <td>940</td>
      <td>299</td>
      <td>1045</td>
      <td>2416</td>
      <td>1614</td>
      <td>902</td>
      <td>1298</td>
      <td>0</td>
      <td>1546</td>
      <td>334</td>
      <td>261</td>
      <td>424</td>
      <td>36</td>
      <td>249</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>2262</td>
      <td>973</td>
      <td>769</td>
      <td>1035</td>
      <td>81</td>
      <td>792</td>
      <td>34</td>
      <td>8</td>
      <td>25</td>
      <td>1826</td>
      <td>364</td>
      <td>406</td>
      <td>82</td>
      <td>36</td>
      <td>2565</td>
      <td>5041</td>
      <td>2597</td>
      <td>5685</td>
      <td>0.439024</td>
      <td>99.619965</td>
      <td>103.193256</td>
      <td>-3.573291</td>
      <td>72.016722</td>
      <td>15.822143</td>
      <td>24.879655</td>
      <td>0.434381</td>
      <td>0.456816</td>
      <td>0.332266</td>
      <td>0.725560</td>
      <td>64.388811</td>
      <td>12.006642</td>
      <td>35.202452</td>
      <td>0.478850</td>
      <td>0.508828</td>
      <td>0.318085</td>
      <td>0.744557</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612739</td>
      <td>CLE</td>
      <td>2379600</td>
      <td>7585</td>
      <td>6207</td>
      <td>2817</td>
      <td>2187</td>
      <td>1653</td>
      <td>801</td>
      <td>298</td>
      <td>993</td>
      <td>2293</td>
      <td>1362</td>
      <td>557</td>
      <td>920</td>
      <td>1180</td>
      <td>430</td>
      <td>405</td>
      <td>504</td>
      <td>60</td>
      <td>218</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>1902</td>
      <td>1183</td>
      <td>1073</td>
      <td>1247</td>
      <td>126</td>
      <td>597</td>
      <td>53</td>
      <td>4</td>
      <td>21</td>
      <td>1894</td>
      <td>388</td>
      <td>419</td>
      <td>7578</td>
      <td>7594</td>
      <td>7361</td>
      <td>6214</td>
      <td>2689</td>
      <td>2175</td>
      <td>1642</td>
      <td>992</td>
      <td>341</td>
      <td>979</td>
      <td>2198</td>
      <td>1389</td>
      <td>654</td>
      <td>1058</td>
      <td>0</td>
      <td>1123</td>
      <td>333</td>
      <td>375</td>
      <td>516</td>
      <td>33</td>
      <td>293</td>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>1785</td>
      <td>1106</td>
      <td>1050</td>
      <td>1276</td>
      <td>88</td>
      <td>822</td>
      <td>63</td>
      <td>0</td>
      <td>19</td>
      <td>1790</td>
      <td>362</td>
      <td>455</td>
      <td>82</td>
      <td>47</td>
      <td>2348</td>
      <td>5222</td>
      <td>2519</td>
      <td>5406</td>
      <td>0.573171</td>
      <td>100.092373</td>
      <td>96.931788</td>
      <td>3.160584</td>
      <td>71.338084</td>
      <td>10.570071</td>
      <td>28.859857</td>
      <td>0.453842</td>
      <td>0.465964</td>
      <td>0.372035</td>
      <td>0.755830</td>
      <td>68.764814</td>
      <td>13.062944</td>
      <td>28.641032</td>
      <td>0.432733</td>
      <td>0.449636</td>
      <td>0.343750</td>
      <td>0.754943</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612741</td>
      <td>CHI</td>
      <td>2350800</td>
      <td>7834</td>
      <td>6696</td>
      <td>3026</td>
      <td>1983</td>
      <td>1478</td>
      <td>937</td>
      <td>304</td>
      <td>1281</td>
      <td>2338</td>
      <td>1080</td>
      <td>559</td>
      <td>854</td>
      <td>1283</td>
      <td>239</td>
      <td>452</td>
      <td>748</td>
      <td>39</td>
      <td>247</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>2073</td>
      <td>667</td>
      <td>1196</td>
      <td>1822</td>
      <td>114</td>
      <td>714</td>
      <td>86</td>
      <td>5</td>
      <td>18</td>
      <td>1931</td>
      <td>343</td>
      <td>349</td>
      <td>7378</td>
      <td>7385</td>
      <td>7244</td>
      <td>6389</td>
      <td>2752</td>
      <td>1933</td>
      <td>1413</td>
      <td>1015</td>
      <td>327</td>
      <td>1078</td>
      <td>2121</td>
      <td>1204</td>
      <td>540</td>
      <td>903</td>
      <td>0</td>
      <td>1216</td>
      <td>274</td>
      <td>347</td>
      <td>586</td>
      <td>26</td>
      <td>271</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2026</td>
      <td>733</td>
      <td>1003</td>
      <td>1608</td>
      <td>87</td>
      <td>787</td>
      <td>106</td>
      <td>6</td>
      <td>29</td>
      <td>1575</td>
      <td>331</td>
      <td>348</td>
      <td>81</td>
      <td>62</td>
      <td>2425</td>
      <td>5374</td>
      <td>2722</td>
      <td>5759</td>
      <td>0.765432</td>
      <td>106.180537</td>
      <td>98.090724</td>
      <td>8.089812</td>
      <td>78.056384</td>
      <td>12.699919</td>
      <td>26.877202</td>
      <td>0.451912</td>
      <td>0.472652</td>
      <td>0.324440</td>
      <td>0.745335</td>
      <td>72.769127</td>
      <td>13.744076</td>
      <td>26.174678</td>
      <td>0.430740</td>
      <td>0.451247</td>
      <td>0.322167</td>
      <td>0.730988</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1610612742</td>
      <td>DAL</td>
      <td>2327830</td>
      <td>7296</td>
      <td>6577</td>
      <td>2802</td>
      <td>1691</td>
      <td>1281</td>
      <td>1159</td>
      <td>411</td>
      <td>1058</td>
      <td>2132</td>
      <td>1084</td>
      <td>606</td>
      <td>852</td>
      <td>923</td>
      <td>338</td>
      <td>363</td>
      <td>766</td>
      <td>60</td>
      <td>337</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>1625</td>
      <td>849</td>
      <td>1061</td>
      <td>1879</td>
      <td>165</td>
      <td>920</td>
      <td>59</td>
      <td>2</td>
      <td>13</td>
      <td>1496</td>
      <td>304</td>
      <td>449</td>
      <td>7412</td>
      <td>7360</td>
      <td>7797</td>
      <td>6633</td>
      <td>3066</td>
      <td>1811</td>
      <td>1292</td>
      <td>1091</td>
      <td>373</td>
      <td>1252</td>
      <td>2498</td>
      <td>1165</td>
      <td>663</td>
      <td>850</td>
      <td>0</td>
      <td>1397</td>
      <td>309</td>
      <td>368</td>
      <td>619</td>
      <td>41</td>
      <td>321</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>2261</td>
      <td>786</td>
      <td>938</td>
      <td>1555</td>
      <td>110</td>
      <td>915</td>
      <td>42</td>
      <td>5</td>
      <td>19</td>
      <td>1845</td>
      <td>343</td>
      <td>365</td>
      <td>80</td>
      <td>19</td>
      <td>2693</td>
      <td>5542</td>
      <td>2391</td>
      <td>5418</td>
      <td>0.237500</td>
      <td>98.434970</td>
      <td>105.937500</td>
      <td>-7.502530</td>
      <td>73.097679</td>
      <td>15.636805</td>
      <td>22.814355</td>
      <td>0.426030</td>
      <td>0.441307</td>
      <td>0.354616</td>
      <td>0.757540</td>
      <td>75.298913</td>
      <td>14.823370</td>
      <td>24.605978</td>
      <td>0.462234</td>
      <td>0.485926</td>
      <td>0.341888</td>
      <td>0.713418</td>
    </tr>
  </tbody>
</table>

#### Player-level aggregation for each season

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEASON_YEAR</th>
      <th>SEASON_TYPE</th>
      <th>PLAYER_ID</th>
      <th>PLAYER_NAME</th>
      <th>PTS</th>
      <th>FGA</th>
      <th>FG</th>
      <th>FTA</th>
      <th>FT</th>
      <th>FG3A</th>
      <th>FG3</th>
      <th>ON_PTS</th>
      <th>ON_OPP_PTS</th>
      <th>ON_OFF_POSS</th>
      <th>ON_DEF_POSS</th>
      <th>GAMES</th>
      <th>FG2</th>
      <th>FG2A</th>
      <th>ORTG</th>
      <th>DRTG</th>
      <th>NRTG</th>
      <th>FG2A_RATE</th>
      <th>FG3A_RATE</th>
      <th>FTA_RATE</th>
      <th>FG%</th>
      <th>FG2%</th>
      <th>FG3%</th>
      <th>FT%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>100</td>
      <td>Tim Legler</td>
      <td>9</td>
      <td>19</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>139</td>
      <td>8</td>
      <td>3</td>
      <td>13</td>
      <td>100.000000</td>
      <td>107.194245</td>
      <td>-7.194245</td>
      <td>8.724832</td>
      <td>4.026846</td>
      <td>2.684564</td>
      <td>0.157895</td>
      <td>0.230769</td>
      <td>0.000000</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1000</td>
      <td>Shandon Anderson</td>
      <td>540</td>
      <td>408</td>
      <td>212</td>
      <td>148</td>
      <td>109</td>
      <td>26</td>
      <td>7</td>
      <td>2636</td>
      <td>2586</td>
      <td>2448</td>
      <td>2507</td>
      <td>71</td>
      <td>205</td>
      <td>382</td>
      <td>107.679739</td>
      <td>103.151177</td>
      <td>4.528562</td>
      <td>15.604575</td>
      <td>1.062092</td>
      <td>6.045752</td>
      <td>0.519608</td>
      <td>0.536649</td>
      <td>0.269231</td>
      <td>0.736486</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1002</td>
      <td>Reggie Geary</td>
      <td>152</td>
      <td>169</td>
      <td>56</td>
      <td>56</td>
      <td>28</td>
      <td>40</td>
      <td>12</td>
      <td>1265</td>
      <td>1209</td>
      <td>1251</td>
      <td>1300</td>
      <td>62</td>
      <td>44</td>
      <td>129</td>
      <td>101.119105</td>
      <td>93.000000</td>
      <td>8.119105</td>
      <td>10.311751</td>
      <td>3.197442</td>
      <td>4.476419</td>
      <td>0.331361</td>
      <td>0.341085</td>
      <td>0.300000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1003</td>
      <td>Drew Barry</td>
      <td>56</td>
      <td>37</td>
      <td>18</td>
      <td>13</td>
      <td>11</td>
      <td>20</td>
      <td>9</td>
      <td>562</td>
      <td>511</td>
      <td>489</td>
      <td>493</td>
      <td>26</td>
      <td>9</td>
      <td>17</td>
      <td>114.928425</td>
      <td>103.651116</td>
      <td>11.277310</td>
      <td>3.476483</td>
      <td>4.089980</td>
      <td>2.658487</td>
      <td>0.486486</td>
      <td>0.529412</td>
      <td>0.450000</td>
      <td>0.846154</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-98</td>
      <td>Regular+Season</td>
      <td>1005</td>
      <td>Walt Williams</td>
      <td>608</td>
      <td>544</td>
      <td>210</td>
      <td>125</td>
      <td>108</td>
      <td>219</td>
      <td>80</td>
      <td>2805</td>
      <td>3019</td>
      <td>2877</td>
      <td>2874</td>
      <td>59</td>
      <td>130</td>
      <td>325</td>
      <td>97.497393</td>
      <td>105.045233</td>
      <td>-7.547840</td>
      <td>11.296489</td>
      <td>7.612096</td>
      <td>4.344804</td>
      <td>0.386029</td>
      <td>0.400000</td>
      <td>0.365297</td>
      <td>0.864000</td>
    </tr>
  </tbody>
</table>

## Current State of the 2020-21 NBA Season
Let us first observe the state of the NBA as of February 11th, 2021. The high-level metrics used to evaluate teams are offensive, defensive, and net rating, which are simply point scored, points allowed, and point differential per 100 possessions. For offensive rating, the higher the figure the better and for defensive rating, the lower the figure the better.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://chart-studio.plotly.com/~achen719/11.embed" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://chart-studio.plotly.com/~achen719/13.embed" height="525" width="100%"></iframe>

# Mean Regression
## Reducing Noise in Defensive Rating
One should always question how sustainable early season trends are. The first area we will examine is defensive rating. If we look at the current defensive rating rankings and the changes from the previous season, we can see that there are some newcomers to the top of the pack. Our methodology will be to completely regress certain metrics to their mean; that is, we will be **replacing those metrics with the league average**.

The two areas that are prime for complete regression to the mean are opponent free throw percentage and opponent three-point percentage. The rationale are as follows:
* **Free throws** occur during a stoppage of play, so a defense has no way to defend the actual free throw attempt. Therefore, the only way to defend free throws is to prevent them entirely. Any deviations from league average in a team's opponent free throw percentage is almost entirely luck.
* **Three-point shooting** in and of itself is extremely noisey, due in large part to their difficulty. This when compounded with the fact that a team's opponent three-point shooting metrics consist of a limited sample of each opposing team, makes opponent three-point shooting also prime for complete regression to the mean.

Below is the function used to completely regress opponent free-throw and three-point percentage to league average. We simply find the difference between a team's opponent free-throw/three-point percentage and the league average, multiply it by the *rate* of opponent free-throw/three-point attempts they allow, and either add or subtract that to the unadjusted defensive rating depending on which direction the first difference was done.

```python
def drtg_adj(df):
    df['DRTG_FT_DIFF'] = df['OPP_FTA_RATE'] * (df['FT%_LG_AVG'] - df['OPP_FT%'])
    df['DRTG_FT_ADJ'] = df['DRTG'] - df['DRTG_FT_DIFF']
    
    df['DRTG_FG3_DIFF'] = df['OPP_FG3A_RATE'] * (df['FG3%_LG_AVG'] - df['OPP_FG3%'])
    df['DRTG_FG3_ADJ'] = df['DRTG'] - df['DRTG_FG3_DIFF']
    
    df['DRTG_FT_FG3_DIFF'] = df['DRTG_FT_DIFF'] + df['DRTG_FG3_DIFF']
    df['DRTG_FT_FG3_ADJ'] = df['DRTG'] + (df['DRTG_FT_DIFF'] + df['DRTG_FG3_DIFF'])
```

We take a look at the distribution of the adjustments made by completely regressing opponent free throw percentage and three point percentage completely to the mean.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://chart-studio.plotly.com/~achen719/15.embed" height="525" width="100%"></iframe>

We see that the distributions of the changes in defensive rating resulting from the oponent FG3% and FT% adjustments are normal and centered at 0 and contains some outliers. This is more or less what we want since these adjustments shouldn't completely change each team's defensive ratings.

Next, we look at how these adjustments effect current defensive ratings for the 2020-21 season. A quick note: the changes in defensive rating from opponent FT% and FG3% adjustments are such that a positive number means the defensive rating worsens and a negative number means the defensive rating improves.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/17.embed" height="525" width="100%"></iframe>

For example, these adjustments tell us that the **New York Knicks (NYK)** are very likely to see their defensive ratings experience *some* regression to the mean. The reasons (shown below) are that that the Knicks currently allow three-point attempts at a rate much higher than league average AND their opponents are shooting well below league average on those attempts. The same also applies to their opponents' free throw shooting, but to a much lesser extent.

It should be noted that this does not mean the Knicks' defensive performance has been complete luck or that they are a bad defensive team, but rather that randomness and variance are having an outsized effect on their results. In fact, if we consider that they have been in the bottom third of defensive rankings for the last four years, this level of improvement should be commended.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TEAM_ABBREVIATION</th>
      <th>OPP_FG3%</th>
      <th>FG3%_LG_AVG</th>
      <th>OPP_FG3A_RATE</th>
      <th>FG3A_RATE_LG_AVG</th>
      <th>OPP_FT%</th>
      <th>FT%_LG_AVG</th>
      <th>OPP_FTA_RATE</th>
      <th>FTA_RATE_LG_AVG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>698</th>
      <td>NYK</td>
      <td>0.321757</td>
      <td>0.369294</td>
      <td>39.128697</td>
      <td>34.632741</td>
      <td>0.75382</td>
      <td>0.773807</td>
      <td>23.541167</td>
      <td>21.799262</td>
    </tr>
  </tbody>
</table>

Next, we look at how opponent FT% and FG3% adjusted defensive rating holds up as the regular season progresses. For each of past seasons, we will find the correlation between the to-date-defensive ratings at different points of the season and rest-of-season figures.

![png](_posts/Early Season Trends/Correlation graph.png)

    <ggplot: (-9223371856543334016)>

We can see that in terms of correlation with defensive performance for the remainder of the season, using adjusted defensive ratings outperforms unadjusted defensive ratings at *every* point in the season. Of course, these are not drastic improvements in correlation and unadjusted defensive rating is already very correlated to the rest-of-season defensive rating, but the improvements are consistent at every level.

## Mean Regression for WIN% and Shooting Percentages
An important concept in applied mathematics is the idea that the **observed result or what actually happens** has two components: the **true representation of what is being measured** and **noise**. In the NBA, when a team wins or loses a game or when a player makes or misses a shot, there is a fair bit of randomness and luck involved. 

\\[Observed \; Result = True \; Talent + Noise\\]

This means that even if a team has the ability to win 80% of their games its *actual* winning percentage will be likely be different due to randomness. Another way to look at is if we observe a team that has won 8 of its first games, what do we make of this team's ability to win its *remaining* games? Are we observing a 60% win team that has gotten rather lucky or is this a 90% win team that has gotten a bit unlucky?
### Formula for Mean Regression
For this notebook, we will be limiting our scope to the **binary variables** win percentage and shooting percentage. We will first define the method for using mean regression to estimate true talent with an example using win percentage,

\\[ True \; WIN\%= \frac{WIN + r \cdot WIN\%_{LG \: AVG}}{GP + r}\\]

where \\(r\\) is the **regression constant** and \\(GP\\) is the **total number of games played**.

If we return to the previous example of the team that won 8 of its first 10 games and assume that the regression constant for win percentage is 12, then the estimated true win percentage for that team would be \\(\frac{8 + 12 \cdot 0.5}{10 + 12} = \frac{14}{22} \approx 63.64\% \\) as opposed to the observed 80% win percentage.

The intuition behind this is that until enough evidence is provided it is safer to assume that someone or something is closer to being average than exceptional. What the above formulation does is take into account that teams on average win only half of their games. The regression constant tells us how *much* weight we should place on this prior knowledge, and at what point observed results outweigh this assumption.
### Finding the regression constant
The beauty of this method lies in the ease of its application *and* that the regression constant, the weight we put on the league average, stays the same regardless of how much data we have. All we need now is the regression constant.

Based on what we stated above about observed result and true talent, we have that

\\[Var(Obs.) = Var(True) + Var(Noise) \iff Var(True) = Var(Obs.) - Var(Noise) \\]

First, we note that, in practice, we only have two things: the observed result (the data) and assumptions on the true nature of the data. Using the former, allows us to obtain the variance of the actual results and the latter, gives us the variance of the noise.

In the case of winning games and making shots, we essentially have a series of success vs. failure situations or a binomial distribution. A binomial distribution has two parameters **n**, the number of trials, and **p**, the probability of success (or failure). In our case, **p** is the likelihood of winning a game or making a shot and **n** is the number of games in the regular season or the number of shot attempts taken in the regular season. Using the assumption of a binomial distribution for a team's ability to win games and make shots, we have that

\\[Var(Noise) = \frac{p \cdot (1 - p)}{n}\\]

The final step is to find the number of trials or the **regression constant** , **r**, such that the variance of true talent *equal* to the random binomial variance:

\\[Var(True \; Talent) = \frac{p \cdot (1 - p)}{r} \iff r = \frac{p \cdot (1 - p)}{Var(True \; Talent)}\\]

```python
# Create list of the last n_years from year
def season_list(year, n_years):
    if year - n_years  + 1>= 1997:
        return [str(year) + '-' + str((year + 1))[2:] for year in range(year - n_years + 1, year + 1)]
    else:
        return [str(year) + '-' + str((year + 1))[2:] for year in range(1997, 1997 + n_years)]

# Filter the dataframe for the last n_years
def group_by_season(df, season_list):
    return df[df['SEASON_YEAR'].isin(season_list)]

# Find the regression to the mean constant
def find_reg_const(df, metric):
    assert metric in ['FG%', 'FT%', 'FG3%', 'FG2%', 'WIN%'], 'The argument "var" must be one of FG%, FT%, FG3%, FG2%, or WIN%'
    denom_map = {'FG%': 'FGA', 
                 'FT%': 'FTA',
                 'FG3%': 'FG3A',
                 'FG2%': 'FG2A',
                 'WIN%': 'GAMES'}
    denom = denom_map[metric]
    metric_mean, denom_mean = df[[metric, denom]].mean()
    metric_var = df[metric].var()
    true_var = metric_var - metric_mean * (1 - metric_mean) / denom_mean
    return metric_mean * (1 - metric_mean) / true_var, metric_mean

# As-at metric vs. metric with mean regression
def create_mean_reg_df(df, metric):
# PROBLEM WITH GAMES
    assert metric in ['FG%', 'FT%', 'FG3%', 'FG2%', 'WIN%'], 'The argument "var" must be one of FG%, FT%, FG3%, FG2%, or WIN%'
    metric_map = {'FG%': ['FG', 'FGA'], 
                  'FT%': ['FT','FTA'],
                  'FG3%': ['FG3', 'FG3A'],
                  'FG2%': ['FG2', 'FG2A'],
                  'WIN%': ['WIN', 'GAMES']}
    # Regression constant and 
    reg_const, metric_mean = find_reg_const(df, metric)
    if metric != 'WIN%':
        cols = metric_map[metric] + ['GAMES']
    else:
        cols = metric_map[metric]
    df_ret = df_team_base[['SEASON_YEAR', 'TEAM_ABBREVIATION'] + cols]
    # Cumulative sums (via groupby) for variables used to calculate the metric
    tots = [n + '_TOT' for n in cols]
    df_ret[tots] = df_ret.groupby(['SEASON_YEAR', 'TEAM_ABBREVIATION']).cumsum()
    # To-date metric
    df_ret[metric + '_ASAT'] = df_ret[metric_map[metric][0] + '_TOT'] / df_ret[metric_map[metric][1] + '_TOT']
    # Calculate mean-regressed metric
    df_ret[metric + '_MEAN_REG'] = (df_ret[metric_map[metric][0] + '_TOT'] + reg_const * metric_mean) / (df_ret[metric_map[metric][1] + '_TOT'] + reg_const)
    return df_ret
```

Above are functions we will use to aid in finding the mean regression constants and applying them to the actual data. Below is the step-by-step process of finding and applying the desired regression coefficient:

    (1)  Aggregate data to the team-level for each available season, excluding the current unfinished season.
    (2)  Find the observed variance and then the random binomial variance using the population mean as **p** and the average number of trials as **n**.
    (3)  Calculate the true variance.
    (4)  Find the regression constant using the true variance and population mean.
    (5)  Apply mean regression at each point in the season for each team.
   
Below are the regression constants for WIN%, FG%, FT%, FG3%, and FG2%, respectively.

```python
feats = ['WIN%', 'FG%', 'FT%', 'FG3%', 'FG2%']
reg_const_dict = {}
for feat in feats:
    reg_const_dict[feat] = find_reg_const(df_team[df_team['SEASON_YEAR'] != '2020-21'], feat)[0]
reg_const_dict
```


    {'WIN%': 12.357970862876234,
     'FG%': 1119.7621837238057,
     'FT%': 226.14498428888763,
     'FG3%': 816.8495686046931,
     'FG2%': 422.8110927433304}

One possible issue is the changing state of the league. The most obvious example of this is the three-pointers attempted in the 1997-98 season pales in comparison to the number of attempts in the 2019-20 season. To adjust for this, we can take the last ten years worth of data rather than the entire dataset.

```python
feats = ['WIN%', 'FG%', 'FT%', 'FG3%', 'FG2%']
reg_const_dict = {}
for feat in feats:
    reg_const_dict[feat] = find_reg_const(group_by_season(df_team, season_list(2019, 10)), feat)[0]
reg_const_dict
```

    {'WIN%': 12.379706720484185,
     'FG%': 1280.9413301838633,
     'FT%': 223.9953682111986,
     'FG3%': 1110.7303535649298,
     'FG2%': 447.8263752377516}

Ten years is admittedly a semi-arbitrary cutoff point, but we see that most of the regression coefficients more or less stay the same except for three-point percentage. Remember that the units for the shooting percentages are their respective shot attempts and the unit for win percentage is games played.

Two-pointers and free throws also have relatively low regression constants, relative to the average number of attempts in a game. Three-pointers take quite awhile to stabilise, which makes sense due to their volatility.

The most surprising result is that it only requires about 12 and a half games -- roughly 1/7th of a typical regular season -- for winning percentage to stabilise, Essentially, only about 12 games worth of data are required to begin inferring a team's true ability to win games. Obviously, one will always find exceptions to such a general rule, but even then this is a very low amount. I believe this may be because games last so long -- each team has the ball for, on average, around 100 possessions a game -- and this is something that is definitely worth further investigating.
### Plotting mean-regressed results
Now we observe the effects of mean regression on the 2019-20 season. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/21.embed" height="525" width="100%"></iframe>
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/23.embed" height="525" width="100%"></iframe>
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/25.embed" height="525" width="100%"></iframe>
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/27.embed" height="525" width="100%"></iframe>

We can see that the mean regression is reactive to, but also tempers early-season variations in win percentage and shooting percentage. Also notice that eventually, the mean-regressed value and the actual to-date values become very, very close because when enough evidence is presented *actual* results matter more.

This is exactly what we want! We are looking for a way to handle *early*-season trends when there isn't enough data. For those interested, the current results for the 2020-21 season are also included:

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/44.embed" height="525" width="100%"></iframe>
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/46.embed" height="525" width="100%"></iframe>
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/48.embed" height="525" width="100%"></iframe>
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/50.embed" height="525" width="100%"></iframe>
## Using Empircal Bayes Estimation for Shooting Percentages
What if, instead of just a best estimate, we wanted a range of probable results?

A possible means to achieve this is to use the beta distribution. An intuitive description of the beta distribution is that is a probability distribution of probabilities and a versatile way to represent outcomes for percentages and proportions. This gives us a method for modeling a player's *true* ability to shoot and represent it in using a range of probable values.

The methodology for doing so is as follows:
1. We fit our population of players' regular season shooting percentages to a beta distribution in order to estimate the parameters for the **prior** distribution.
2. We use a player's **made and missed shots** to update the alpha and beta parameters for the **posterior** beta distribution at each point in the season.
3. We use the updated distribution to find the upper and lower bounds for the 68% and 95% credible intervals at each point in the season.
4. We find the expectation (mean) of the updated distribution.

In a bit more understandable terms, we use the entire dataset to establish *assumptions* (ie. the prior distribution) of how well shooters generally perform through the course of a season. At the beginning of each season, each player begins with this base assumption of how well they're expected to perform. Because these assumptions are based on the entire dataset, the range of probable shooting percentages is pretty wide at the onset of the season. As players play more games and they take more shots, the results of these shots are used to update what we believe is a representation of their true ability to make a shot.

Like the mean regression method used previously, doing this tempers early season fluctuations and gives a *range* of probable outcomes going forward.

### Estimating a prior distribution
First, we empirically (using the available data) estimate the prior beta distributions for each of the shooting percentages.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/Early Season Trends/Unfiltered distribution plots.png" alt="">


![png](_posts/Early Season Trends/Unfiltered distribution plots.png)

    <ggplot: (-9223371856543409432)>

The distributions for three-point and free-throw percentage have spike at 0 and 1 because there are quite a number of players simply did not or did not take many three-pointers/free-throws through the course of the season. There are also players that played a very limited number of games or minutes in a season.

We will simply limit the dataset to players with at least one attempt per game and at least twenty games played in the season.

![png](_posts/Early Season Trends/Filtered distribution plots.png)

    <ggplot: (-9223371856543342296)>

By filtering the data a bit, we can see that three-point and two-point percentage are normally skewed, whereas free-throw percentage is skewed to the right. Next, we will fit the data to beta distributions.


```python
fg3_player = df_player[(df_player['FG3A']/df_player['GAMES'] >= 1) & 
                       (df_player['GAMES'] >= 20) & 
                       (df_player['SEASON_YEAR'] != '2020-21')]
```


```python
fg2_player = df_player[(df_player['FG2A']/df_player['GAMES'] >= 1) & 
                       (df_player['GAMES'] >= 20) & 
                       (df_player['SEASON_YEAR'] != '2020-21')]
```


```python
ft_player = df_player[(df_player['FTA']/df_player['GAMES'] >= 1) & 
                      (df_player['GAMES'] >= 20) & 
                      (df_player['SEASON_YEAR'] != '2020-21')]
```

A quick note on **scipy.stats.beta.fit()**: if the location and scale variables aren't set ahead of time, the function fits a *generalized* beta distribution -- one that has been shifted and stretched beyond the interval $[0, 1]$.


```python
a0_fg3, b0_fg3, loc_fg3, scale_fg3 = stats.beta.fit(fg3_player['FG3%'], floc = 0, fscale = 1.)
a0_fg3, b0_fg3, loc_fg3, scale_fg3
```




    (28.520941901347765, 52.94671022688583, 0, 1.0)




```python
a0_fg2, b0_fg2, loc_fg2, scale_fg2 = stats.beta.fit(fg2_player['FG2%'], floc = 0, fscale = 1.)
a0_fg2, b0_fg2, loc_fg2, scale_fg2
```




    (29.74959053829624, 32.97187867341685, 0, 1.0)




```python
a0_ft, b0_ft, loc_ft, scale_ft = stats.beta.fit(ft_player['FT%'], floc = 0, fscale = 1.)
a0_ft, b0_ft, loc_ft, scale_ft
```




    (13.712912717517055, 4.706635476245001, 0, 1.0)



We can see that the empirically estimated beta distributions fit pretty well, albeit not perfectly.
### Using the estimated beta distributions as priors for individual estimates
Using the estimated beta distributions for shooting percentages above, we will now begin to make individual estimates for players. How we do so is actually quite simple. The prior beta distributions are updated with the actual results. More specifically,

\\[Beta(\alpha_0, \beta_0) \rightarrow Beta(\alpha_0 + made \: shots, \beta_0 + missed \: shots)\\]

where \\(\alpha_0\\) and \\(\beta_0\\) are the alphas and betas attained above. The proof for this is supplied below for those mathematically inclined is [here](https://en.wikipedia.org/wiki/Conjugate_prior#Example).

The beauty of this method is the simplicity of the implementation, but also the robustness of the math behind it.
### Plotting expected shooting percentages with credible intervals

First, we look at the top of the leaderboards for two-point and three-point percentage for the 2020-21 season.
#### 2020-21 Three-point percentage leaders
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEASON_YEAR</th>
      <th>PLAYER_NAME</th>
      <th>FG3A</th>
      <th>FG3%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11148</th>
      <td>2020-21</td>
      <td>Tony Snell</td>
      <td>26</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>11019</th>
      <td>2020-21</td>
      <td>Mason Jones</td>
      <td>30</td>
      <td>0.533333</td>
    </tr>
    <tr>
      <th>11116</th>
      <td>2020-21</td>
      <td>Jeremy Lamb</td>
      <td>50</td>
      <td>0.520000</td>
    </tr>
    <tr>
      <th>11155</th>
      <td>2020-21</td>
      <td>Seth Curry</td>
      <td>82</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>11136</th>
      <td>2020-21</td>
      <td>Gorgui Dieng</td>
      <td>38</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>11168</th>
      <td>2020-21</td>
      <td>Joe Harris</td>
      <td>181</td>
      <td>0.491713</td>
    </tr>
    <tr>
      <th>10780</th>
      <td>2020-21</td>
      <td>Bryn Forbes</td>
      <td>104</td>
      <td>0.490385</td>
    </tr>
    <tr>
      <th>11015</th>
      <td>2020-21</td>
      <td>Desmond Bane</td>
      <td>85</td>
      <td>0.482353</td>
    </tr>
    <tr>
      <th>10736</th>
      <td>2020-21</td>
      <td>Bobby Portis</td>
      <td>52</td>
      <td>0.480769</td>
    </tr>
    <tr>
      <th>11081</th>
      <td>2020-21</td>
      <td>Paul George</td>
      <td>157</td>
      <td>0.477707</td>
    </tr>
  </tbody>
</table>
#### 2020-21 Two-point percentage leaders
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEASON_YEAR</th>
      <th>PLAYER_NAME</th>
      <th>FG2A</th>
      <th>FG2%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10802</th>
      <td>2020-21</td>
      <td>Jarrett Allen</td>
      <td>153</td>
      <td>0.673203</td>
    </tr>
    <tr>
      <th>10727</th>
      <td>2020-21</td>
      <td>Richaun Holmes</td>
      <td>188</td>
      <td>0.664894</td>
    </tr>
    <tr>
      <th>10723</th>
      <td>2020-21</td>
      <td>Montrezl Harrell</td>
      <td>218</td>
      <td>0.660550</td>
    </tr>
    <tr>
      <th>10795</th>
      <td>2020-21</td>
      <td>Lauri Markkanen</td>
      <td>84</td>
      <td>0.654762</td>
    </tr>
    <tr>
      <th>10866</th>
      <td>2020-21</td>
      <td>Mitchell Robinson</td>
      <td>150</td>
      <td>0.653333</td>
    </tr>
    <tr>
      <th>11151</th>
      <td>2020-21</td>
      <td>Giannis Antetokounmpo</td>
      <td>340</td>
      <td>0.644118</td>
    </tr>
    <tr>
      <th>11044</th>
      <td>2020-21</td>
      <td>Thaddeus Young</td>
      <td>152</td>
      <td>0.638158</td>
    </tr>
    <tr>
      <th>11169</th>
      <td>2020-21</td>
      <td>Doug McDermott</td>
      <td>143</td>
      <td>0.636364</td>
    </tr>
    <tr>
      <th>11145</th>
      <td>2020-21</td>
      <td>Rudy Gobert</td>
      <td>213</td>
      <td>0.619718</td>
    </tr>
    <tr>
      <th>10798</th>
      <td>2020-21</td>
      <td>John Collins</td>
      <td>227</td>
      <td>0.616740</td>
    </tr>
    <tr>
      <th>10738</th>
      <td>2020-21</td>
      <td>Christian Wood</td>
      <td>191</td>
      <td>0.612565</td>
    </tr>
    <tr>
      <th>11139</th>
      <td>2020-21</td>
      <td>Mason Plumlee</td>
      <td>157</td>
      <td>0.611465</td>
    </tr>
    <tr>
      <th>10841</th>
      <td>2020-21</td>
      <td>Mikal Bridges</td>
      <td>126</td>
      <td>0.611111</td>
    </tr>
    <tr>
      <th>10908</th>
      <td>2020-21</td>
      <td>Zion Williamson</td>
      <td>340</td>
      <td>0.608824</td>
    </tr>
    <tr>
      <th>11186</th>
      <td>2020-21</td>
      <td>Nikola Jokic</td>
      <td>354</td>
      <td>0.607345</td>
    </tr>
  </tbody>
</table>

We want to explore the difference in the number of shot attempts effects the credible intervals, so for three-point percentage we will look at Seth Curry and Joe Harris, two elite shooters that have about an 100 attempt difference between them. For two-point percentage, we will look at the difference in expected performances between Lauri Markkanen and Giannis Antetokounmpo. Note that Giannis has taken roughly four times the amount of two-pointers that Markkanen has.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/33.embed" height="525" width="100%"></iframe>
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/35.embed" height="525" width="100%"></iframe>
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/29.embed" height="525" width="100%"></iframe>
 
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~achen719/31.embed" height="525" width="100%"></iframe>

For both pairs, we see that the players (Joe Harris and Giannis Antetokounmpo) with more attempts have tighter credible intervals as of their latest performances and their actual performances are closer to their expected performance.

## Conclusion
I am rather satisfied with the results above because they help determine how confident we should be in a team or player's current performance going forward. There are, of course, many ways to improve these methods even further. Below, I will go over possible ways to further improve the methods used above.
### Complete Regression to the Mean for Opponent FG3% and FT%
Complete regression to the league average for opponent free throw percentage is sound and I do not see any meaningful improvements to this. However, there is an avenue in which we look at the *quality* of three-point attempts when regressing opponent three-point percentage.

When assessing the quality of a three-point shot, what we should pay attention to are (1) the location of the shot, specifically if it is an above-the-break or corner attempt (2) how good the person taking the shot is at making that shot and (3) how closely defended that shot is. Second Spectrum has a decent amount of publicly available tracking data on the official NBA stats website, which I will be exploring in a later notebook.
### Regressing Binary Variables to the Mean
The method of mean regression for binary variables, could also be improved with more granularity. The location of a shot is massively important in determining the likelihood of it going in. The issue with delineating shots even further is simply that the amount of data we have for each "bin" of shots decreases. It'll be important to find the proper groupings such that each group has enough data *and* each group is actually useful in determining in evaluating teams and players.

Tracking data could potentially be incredibly useful, but because tracking data is only available from the 2013-14 season and on, we do have a more limited sample. Thorough exploration on the significance of each tracking metric is required.

Furthermore, handling the changing state of the NBA could be handled with a bit more care. Choosing to use the last ten years, as we did so above, was kind of arbitrary (not completely) and a more careful selection could prove to be beneficial.
### Empirical Bayes Estimation
The methods for improvement mentioned above also apply to empirical Bayes estimation. Another route for improvement involves the prior beta distribution used for the estimates. 

The parameters for the prior distribution were found by fitting to the entire dataset of players (with some selection, of course). But, is it really fair to compare players from 20 years ago to now? Should we be lumping big men with guards? Should we incorporate a player's performances prior to the season and if so, how? These questions (and more) should all be explored and a more informed prior expectation will likely lead to improved and more useful results.

