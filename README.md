# Football Odds Prediction
# Table of contents
1. [Introduction](#Introduction)
2. [Model Explanation](#Model_Explanation)
3. [Code Structure](#Code_Structure)
4. [Data](#Data)
5. [Methodology](#Methodology)
6. [Results](#Results)
7. [Improvements](#Improvements)

## Introduction <a id="Introduction"></a>
The aim of this project is to analyse historical football results between a pool of teams, and infer winning probabilities between any pair.

We implement the Double Poisson model described by [Dixon and Coles (1997)](https://www.ajbuckeconbikesail.net/wkpapers/Airports/MVPoisson/soccer_betting.pdf).
## Model Explanation <a id="Model_Explanation"></a>
Aside from the paper linked above, I give a brief explanation of the model in [this Linkedin article](https://www.linkedin.com/pulse/ranking-epl-football-teams-dylan-zammit%3FtrackingId=MUHlwKLNTaW8LWBOGalsww%253D%253D/?trackingId=MUHlwKLNTaW8LWBOGalsww%3D%3D).

Each team `i` is given an `attack` and `defence` score denoted by `a_i` and `d_i` respectively. The `home advantage` is denoted by `K`.

The main assumption of the model is that the goals scored by both teams are Poisson-distributed and independent of each other. Matches are also assumed to be independent. The rate parameter of the Poisson distribution of the home team `i` against the away team `j` is

```rate_i = a_i * d_j * K```

whereas the rate parameter of the away team is

```rate_j = a_j * d_i```.
The probability of such two teams geting a score of `x-y` is thus

```Pr(X=x, Y=y) = x ^ rate_i * exp(-rate_i) * y ^ rate_j * exp(-rate_j) / CONSTANT```

Given a set of `N` matches we can find the optimal `a_i`, `d_i` and `K` for each team by maximising the likelihood

```L(Data) = PRODUCT_ij { Pr(X_i=x_ij, Y_j=y_ij) }```

In order to give higher importance to recent games, an exponential decay is performed on the matches within the likelihood based on the distance in days of the match happening. The decay function is given by `decay(t) = exp(-zeta * x)` to get.

```L(Data) = PRODUCT_ij { [ Pr(X_i=x_ij, Y_j=y_ij) ] ^ decay(t_max - t) }```

## Code Structure <a id="Code_Structure"></a>
To install and be able to use locally, just go to the root directory of the project, containing the `setup.py` file and run `pip install .`.
### Analysis
In this folder we have notebooks with EDA and model evaluation strategies. This folder is not meant to be productionised.
### football_odds
This is the main package containing the utils directory with common methods and classes. This includes the `MarketOdds` class. Below is an example snippet of how this can be used.
```python
from football_odds.utils.odds_compiler import MarketOdds

mo = MarketOdds(
    home_score=(1.2, 0.7),
    away_score=(1.1, 0.9),
    home_adv=1.2,
)

print(mo)
```

<details>
<summary>Output</summary>

```
1x2: (0.48992558472222425, 0.2883851893358551, 0.22168922594192064)

Half Time 1x2: (0.3620983599084602, 0.45043119817708993, 0.1874704419144499)

Correct Score 0-1: 0.09755248246238979
Correct Score 0-2: 0.03755770574802006
Correct Score 0-3: 0.009639811141991816
Correct Score 1-0: 0.1641922302224119
Correct Score 1-2: 0.048674786649434004
Correct Score 1-3: 0.012493195240021394
Correct Score 2-0: 0.10639656518412291
Correct Score 2-1: 0.08192535519177464
Correct Score 2-3: 0.008095590515533864
Correct Score 3-0: 0.04596331615954109
Correct Score 3-1: 0.035391753442846646
Correct Score 3-2: 0.013625825075495958

Over/Under 1.5: (0.6115637516497572, 0.3884362483502428)
Over/Under 2.5: (0.3411814634463568, 0.6588185365536432)
Over/Under 3.5: (0.15497819430361537, 0.8450218056963846)

```
</details>

The `models.py` script contains the class that will fit the model on the specified data, and return test outputs.  After training the model, we can also `save` the object as a pickle, so we can load it later for reuse.

```python
import pandas as pd
from football_odds.models import DoublePoisson, save, load

# required_columns: fixture_date, home_team_name, away_team_name, goals_home, goals_away
df = pd.read_csv('file/containing/matches/played.csv')

dp = DoublePoisson()
dp.fit(df)

model_pkl = 'path/to/model.pkl' 
save(dp, model_pkl)

dp_copy = load(model_pkl)
```
### API
We create a simple API with one `GET` endpoint. This endpoint is in the format of 
```http://127.0.0.1:8000/MATCH_ODDS/{home_team}/{away_team}```

One such example is
```http://127.0.0.1:8000/MATCH_ODDS/Arsenal/Leicester```

This method hits the `DoublePoisson.test(home_team, away_team)` method, that returns a `MarketOdds` objects. This class compiles all relevant probabilities based on the attacking/defensive scores of the home/away team along with the home advantage. For this particular example, only the match outcome probabilities are relevant. 
## Data <a id="Data"></a>
We obtain historical football results of
* England Premier League
* from season starting 2010 to 
* November 2023 

The data was obtained from [API-FOOTBALL](https://www.api-football.com/) and is store them in a local QuestDB instance running on docker. The relevant data obtained in our case is:
* Home team name
* Away team name
* Home goals scored
* Away goals scored
* Fixture Date
## Methodology <a id="Methodology"></a>
### Cleaning
Under `analysis/Pre-Match Analysis.ipynb`, we explore the structure and nature of the data. We first notice that some games erroneously have NULL values under the home/away goals. These are filtered out when querying.

### Poisson-Distributed goals
The main assumption of the Dixon & Cole model is the Poisson distribution assumption of the goals. Upon visual inspection of the histogram, this assumption makes sense. Running the Pearson Chi-square test on the distribution of the homa/away goals against the Poisson, we get high p-values indicating that modelling using the Poisson distribution is adequate.

### Independent home/away goals
Another feature implemented by Dixon & Cole is that low-goal results are dependent. We however test for independence between the home and away goals, again using the Chi-square test for independence. This resulted in a high p-value, meaning that the distributions are statistically independent. This simplified are model considerably. 

### Home Advantage
The summary statistics indicate that the home team scores more goals on average than the away team, indicating that the home team has an inherit advantage.
The Mann-Whitney-U test confirms this suspicion, having a p-value < 0.05 against the alternate hypotheses that the average number of home goals is greater than the away goals. Under the Poisson-distributed-goals assumption, we can use the E-test, which also gives the same results.

### Evaluation
Under `analysis/Evaluation.ipynb`, We use 1 year of data to train the model, from `2021-01-01` until `2021-12-31`.

Since the model output gives the probability of a HOME/DRAW/AWAY win, we can consider this problem as a multiclass classification problem.
We consider the F1, Accuracy, Recall and Precision scores. 
We also calculate the Root Mean Squared Error (RMSE) between the predicted outcome probability and the binary outcome of the predicted event.
We plot the ROC curve and calculate the AUC.

In summary, the following metrics were used:
* F1
* Accuracy
* Precision
* Recall
* RMSE
* AUC

In a **failed** attempt to measure concept drift we subset the test data into 31-day periods, and calculate all metrics for these periods.

## Results <a id="Results"></a>
**NOTE**: Due to the nature of the Poisson model, the probability of a DRAW is lower than a win, and thus the model essentially almost never predicts a DRAW as the most likely outcome. This will skew results negatively when considering the overall weighted F1 score and accuracy.
Below is a summary of the results
* Overall accuracy of 52% (keeping in mind that this is a 3-class classification problem)
* Weighted F1-Score of 61% for HOME guesses and 51% for AWAY guesses
* AUC of 62%
* Concept drift not obvious from the plots, and further analysis is required

## Improvements <a id="Improvements"></a>
* We could apply grid search over the zeta parameter to optimise the decay of old games and choose the best model.
* Better methods to evaluation the optimal concept drift need to be researched and implemented
* Increasing the pool of teams and thus sample size
* Better backtesting framework: simulate betting on the Betfair exchange by backing and laying odds based on the discrepancy of the model using the Kelley Criterion.
