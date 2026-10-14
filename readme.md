# Predicting NFL Team Success


# loaded in data with pandas, The data set contains any metric I thought might be relevant for predicting NFL wins and losses.

``` python
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("~/Downloads/nfl_team_season_stats_2015_2025.csv")
```

\#view the data

``` python
df.head()
```

<div>


|  | season | team | games | wins | losses | ties | win_pct | points_per_game | points_allowed_per_game | point_diff_per_game | ... | sack_rate_for_per_dropback | qb_hits_per_dropback | def_interceptions_per_game | def_tds_per_game | off_tds_allowed_per_game | turnover_margin_per_game | epa_per_play_differential | penalties_per_game | penalty_yards_per_game | fg_pct |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 2015 | ARI | 16 | 13 | 3 | 0 | 0.8125 | 30.5625 | 19.5625 | 11.0000 | ... | 0.0575 | 0.1757 | 1.1875 | 0.2500 | 2.0625 | 0.5625 | 0.2149 | 5.8750 | 47.3750 | 0.9032 |
| 1 | 2015 | ATL | 16 | 8 | 8 | 0 | 0.5000 | 21.1875 | 21.5625 | -0.3750 | ... | 0.0328 | 0.1293 | 0.9375 | 0.0625 | 2.4375 | -0.3125 | -0.0040 | 6.2500 | 60.5625 | 0.8065 |
| 2 | 2015 | BAL | 16 | 5 | 11 | 0 | 0.3125 | 20.5000 | 25.0625 | -4.5625 | ... | 0.0636 | 0.1615 | 0.3750 | 0.0625 | 2.5000 | -0.8750 | -0.0809 | 7.6250 | 72.0625 | 0.8250 |
| 3 | 2015 | BUF | 16 | 8 | 8 | 0 | 0.5000 | 23.6875 | 22.4375 | 1.2500 | ... | 0.0337 | 0.1204 | 1.0625 | 0.1250 | 2.5000 | 0.3125 | 0.0210 | 8.9375 | 78.0625 | 0.8519 |
| 4 | 2015 | CAR | 16 | 15 | 1 | 0 | 0.9375 | 31.2500 | 19.2500 | 12.0000 | ... | 0.0634 | 0.1585 | 1.5000 | 0.2500 | 2.0000 | 1.0625 | 0.2028 | 6.4375 | 55.4375 | 0.8333 |

<p>5 rows × 50 columns</p>
</div>

``` python
df.columns
```

    Index(['season', 'team', 'games', 'wins', 'losses', 'ties', 'win_pct',
           'points_per_game', 'points_allowed_per_game', 'point_diff_per_game',
           'off_epa_per_play', 'off_pass_epa_per_dropback',
           'off_rush_epa_per_carry', 'yards_per_game', 'pass_yards_per_game',
           'rush_yards_per_game', 'yards_per_play', 'yards_per_carry',
           'net_yards_per_dropback', 'completion_pct', 'pass_rate',
           'plays_per_game', 'first_down_pct_per_play', 'turnover_rate_per_play',
           'interceptions_per_dropback', 'sacks_against_per_play',
           'sack_rate_against_per_dropback', 'off_tds_per_game',
           'explosive_pass_20plus_per_game', 'explosive_rush_20plus_per_game',
           'def_epa_per_play_allowed', 'def_pass_epa_per_dropback_allowed',
           'def_rush_epa_per_carry_allowed', 'yards_against_per_game',
           'pass_yards_against_per_game', 'rush_yards_against_per_game',
           'yards_per_play_allowed', 'first_down_pct_allowed_per_play',
           'takeaway_rate_per_play', 'sacks_for_per_play',
           'sack_rate_for_per_dropback', 'qb_hits_per_dropback',
           'def_interceptions_per_game', 'def_tds_per_game',
           'off_tds_allowed_per_game', 'turnover_margin_per_game',
           'epa_per_play_differential', 'penalties_per_game',
           'penalty_yards_per_game', 'fg_pct'],
          dtype='str')

\#decided to drop everything that highly correlates wiith wins so the
when I do my correlation matrix the obvious metrics don’t show up

``` python
df_clean = df.drop(columns=['season', 'team', 'games', 'losses', 'ties', 'win_pct'])
```

\#review the cleaned data

``` python
df_clean.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | wins | points_per_game | points_allowed_per_game | point_diff_per_game | off_epa_per_play | off_pass_epa_per_dropback | off_rush_epa_per_carry | yards_per_game | pass_yards_per_game | rush_yards_per_game | ... | sack_rate_for_per_dropback | qb_hits_per_dropback | def_interceptions_per_game | def_tds_per_game | off_tds_allowed_per_game | turnover_margin_per_game | epa_per_play_differential | penalties_per_game | penalty_yards_per_game | fg_pct |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 13 | 30.5625 | 19.5625 | 11.0000 | 0.1102 | 0.2419 | -0.0614 | 428.1875 | 308.3750 | 119.8125 | ... | 0.0575 | 0.1757 | 1.1875 | 0.2500 | 2.0625 | 0.5625 | 0.2149 | 5.8750 | 47.3750 | 0.9032 |
| 1 | 8 | 21.1875 | 21.5625 | -0.3750 | -0.0006 | 0.1108 | -0.1739 | 401.9375 | 301.5625 | 100.3750 | ... | 0.0328 | 0.1293 | 0.9375 | 0.0625 | 2.4375 | -0.3125 | -0.0040 | 6.2500 | 60.5625 | 0.8065 |
| 2 | 5 | 20.5000 | 25.0625 | -4.5625 | -0.0679 | -0.0266 | -0.1432 | 381.5625 | 289.1875 | 92.3750 | ... | 0.0636 | 0.1615 | 0.3750 | 0.0625 | 2.5000 | -0.8750 | -0.0809 | 7.6250 | 72.0625 | 0.8250 |
| 3 | 8 | 23.6875 | 22.4375 | 1.2500 | 0.0508 | 0.0859 | 0.0159 | 393.0625 | 241.0625 | 152.0000 | ... | 0.0337 | 0.1204 | 1.0625 | 0.1250 | 2.5000 | 0.3125 | 0.0210 | 8.9375 | 78.0625 | 0.8519 |
| 4 | 15 | 31.2500 | 19.2500 | 12.0000 | 0.0875 | 0.1608 | 0.0132 | 402.4375 | 259.8125 | 142.6250 | ... | 0.0634 | 0.1585 | 1.5000 | 0.2500 | 2.0000 | 1.0625 | 0.2028 | 6.4375 | 55.4375 | 0.8333 |

<p>5 rows × 44 columns</p>
</div>

\#used .corr() to find the correlation of each numeric feature’s
correlation to wins

``` python
correlations = df_clean.corr(numeric_only=True)['wins'].sort_values(ascending=False)
```

# stats correlated with winning

``` python
print(correlations.head(10))
```

    wins                         1.000000
    point_diff_per_game          0.896742
    epa_per_play_differential    0.854452
    points_per_game              0.751558
    off_pass_epa_per_dropback    0.696417
    off_epa_per_play             0.682448
    turnover_margin_per_game     0.671634
    off_tds_per_game             0.668792
    takeaway_rate_per_play       0.557068
    first_down_pct_per_play      0.542213
    Name: wins, dtype: float64

# stats correlated with losing

``` python
print(correlations.tail(10))
```

    pass_rate                           -0.356461
    sack_rate_against_per_dropback      -0.405890
    interceptions_per_dropback          -0.413817
    rush_yards_against_per_game         -0.468924
    sacks_against_per_play              -0.510807
    def_epa_per_play_allowed            -0.513783
    off_tds_allowed_per_game            -0.516219
    turnover_rate_per_play              -0.543844
    def_pass_epa_per_dropback_allowed   -0.564816
    points_allowed_per_game             -0.662945
    Name: wins, dtype: float64

\#import new libraries in order to run ridge model

\#dropped more columns that were too obvious and completely skewed the
model I also tried out a random forrest model that I deleted,but I
prefered the ridge model due to the random forrest model having higher
dependance on one to two variables

``` python
X = df_clean.drop(columns=['wins', 'point_diff_per_game','epa_per_play_differential'])
Y = df_clean['wins']
```

# import more libraries for scaling and ridge regression

``` python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
```

\#Scaled the features before fitting the ridge regression model.

``` python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

\#run the ridge regression model

``` python
ridge_model = Ridge(alpha=10.0)
ridge_model.fit(X_scaled, Y)
```


``` python
coefficients = pd.Series(ridge_model.coef_, index=X.columns).sort_values(ascending=False)
```

\#print the top positive and negative drivers of wins

``` python
print("--- Top 5 Positive Drivers of Wins ---")
print(coefficients.head(5).round(3))
print("\n--- Top 5 Negative Drivers of Wins ---")
print(coefficients.tail(5).round(3))
```

    --- Top 5 Positive Drivers of Wins ---
    points_per_game              0.971
    off_tds_per_game             0.359
    off_pass_epa_per_dropback    0.353
    sacks_for_per_play           0.280
    off_tds_allowed_per_game     0.261
    dtype: float64

    --- Top 5 Negative Drivers of Wins ---
    yards_per_carry                  -0.198
    explosive_pass_20plus_per_game   -0.211
    rush_yards_against_per_game      -0.244
    turnover_rate_per_play           -0.285
    points_allowed_per_game          -1.102
    dtype: float64

## Model Evaluation & Performance Metrics

evaluate model predictive accuracy across the dataset using Mean
Absolute Error (MAE) and r squared

``` python
predictions = ridge_model.predict(X_scaled)

mae = mean_absolute_error(Y, predictions)
r2 = r2_score(Y, predictions)

print(f"Mean Absolute Error (MAE): {mae:.2f} wins")
print(f"Coefficient of Determination (R²): {r2:.3f}")
```

    Mean Absolute Error (MAE): 1.01 wins
    Coefficient of Determination (R²): 0.837

## Visualizing Key Statistical Drivers

Next I visualized the top positive and negative drivers of season win
totals.

``` python
import matplotlib.pyplot as plt
import seaborn as sns
top_features = pd.concat([coefficients.head(5), coefficients.tail(5)])

plt.figure(figsize=(10, 5))
colors = ['navy' if x > 0 else 'firebrick' for x in top_features.values]
top_features.plot(kind='barh', color=colors)
plt.title('Top Positive and Negative Drivers of NFL Win Totals (Ridge Regression)')
plt.xlabel('Standardized Coefficient Weight')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-17-output-1.png)

## Evaluating Predicted vs. Actual Wins

Comparing predicted win totals against actual season outcomes highlights
teams that over- or under-performed relative to their stat line.

``` python
results = df[['season', 'team', 'wins']].copy()
results['predicted_wins'] = predictions.round(1)
results['residual'] = (results['wins'] - results['predicted_wins']).round(1)
print("--- Sample Predictions vs. Actual Wins ---")
print(results.head(10))
plt.figure(figsize=(7, 6))
plt.scatter(results['predicted_wins'], results['wins'], alpha=0.6, color='darkblue')
plt.plot([0, 17], [0, 17], color='red', linestyle='--', label='Perfect Prediction Line')
plt.title('Actual Wins vs. Predicted Wins')
plt.xlabel('Predicted Wins')
plt.ylabel('Actual Wins')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
```

    --- Sample Predictions vs. Actual Wins ---
       season team  wins  predicted_wins  residual
    0    2015  ARI    13            12.5       0.5
    1    2015  ATL     8             7.8       0.2
    2    2015  BAL     5             5.4      -0.4
    3    2015  BUF     8             8.9      -0.9
    4    2015  CAR    15            13.7       1.3
    5    2015  CHI     6             6.6      -0.6
    6    2015  CIN    12            12.8      -0.8
    7    2015  CLE     3             3.2      -0.2
    8    2015  DAL     4             4.4      -0.4
    9    2015  DEN    12             9.5       2.5

![](readme_files/figure-commonmark/cell-18-output-2.png)

# download new file I manually ade with just a few teams, I couldn’t get all the stats for the 2026 season

``` python
import pandas as pd
df_2026 = pd.read_csv("2026_NFL_data.csv")
```

# Read and clean the 2026 NFL data file

# I couldn’t figure out how to clean the data to properly run it in the model so I got some help writing this for loop to clean the raw CSV lines before loading them into a DataFrame to be able to finish the last part of my project

``` python
import pandas as pd
import io
import os
file_path = os.path.expanduser("2026_NFL_data.csv")
with open(file_path, "r", encoding="utf-8-sig") as f:
    raw_lines = f.readlines()
clean_lines = []
for line in raw_lines:
    line_str = line.strip()
    # Skip empty lines
    if not line_str or line_str.replace(',', '').strip() == '':
        continue
    if line_str.startswith('"'):
        line_str = line_str.split('",,,')[0].lstrip('"').rstrip('",')
    clean_lines.append(line_str)
```

# 2. Load cleaned lines into Pandas DataFrame

``` python
df2 = pd.read_csv(io.StringIO("\n".join(clean_lines)))
```

# 3. Align features with historical training matrix X

``` python
X_2026 = df2[X.columns]
```

# 4. Scale using the fitted historical scaler

``` python
X_2026_scaled = scaler.transform(X_2026)
```

# 5. Predict total wins and calculate remaining wins

``` python
df2['projected_total_wins'] = ridge_model.predict(X_2026_scaled).round(1)
df2['projected_remaining_wins'] = (df2['projected_total_wins'] - df2['wins']).round(1)
```

# 6. Display formatted standings

``` python
projections = df2[['team', 'wins', 'losses', 'points_per_game', 'points_allowed_per_game', 'projected_total_wins', 'projected_remaining_wins']]\
    .sort_values(by='projected_total_wins', ascending=False)\
    .reset_index(drop=True)

projections
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | team | wins | losses | points_per_game | points_allowed_per_game | projected_total_wins | projected_remaining_wins |
|----|----|----|----|----|----|----|----|
| 0 | CHI | 3 | 1 | 28.00 | 16.25 | 13.6 | 10.6 |
| 1 | KC | 4 | 0 | 29.50 | 19.25 | 13.2 | 9.2 |
| 2 | BUF | 3 | 1 | 31.75 | 26.75 | 10.6 | 7.6 |
| 3 | LAR | 2 | 2 | 21.25 | 20.75 | 9.2 | 7.2 |
| 4 | DET | 2 | 2 | 29.75 | 31.75 | 8.0 | 6.0 |

</div>

# visualize the totals for each of the modeled teams

``` python
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
bars = plt.bar(
    projections['team'], 
    projections['projected_total_wins'], 
    width=0.5
)

plt.title('2026 Projected Total Wins', fontsize=12, fontweight='bold', pad=10)
plt.ylabel('Projected Wins')
plt.ylim(0, 16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-26-output-1.png)
