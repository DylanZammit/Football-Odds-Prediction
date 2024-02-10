"""
This script implements the Double-Poisson Model of
Dixon - Modelling Association Football Scores and Inefficiencies in the Football Betting Market
"""

from football_odds.utils import expon, decay, save, load
from scipy.optimize import minimize, OptimizeResult
from football_odds.utils.connectors import QuestDB
from football_odds.utils.odds_compiler import MarketOdds
from functools import partial
from itertools import chain
import pandas as pd
import numpy as np


def partial_likelihood(
        x,
        attack: dict,
        defence: dict,
        home_adv: float,
) -> float:
    """
    :param x: Row containing information about a matchup
    :param attack: dict of teams with their respective attack scores
    :param defence: dict of teams with their respective defence scores
    :param home_adv: home advantage param
    :return: Gives the likelihood of a particular result between two teams given the parameters
    """
    a_home, d_home = attack[x.home_team_name], defence[x.home_team_name]
    a_away, d_away = attack[x.away_team_name], defence[x.away_team_name]
    lam = a_home * d_away * home_adv
    mu = a_away * d_home

    return expon(x.goals_home, lam) * expon(x.goals_away, mu)


def log_likelihood(
        df_games,
        attack: dict,
        defence: dict,
        home_adv: float,
        zeta: float,
) -> float:
    """
    :param df_games: DataFrame of fixtures containing the home/away teams, result, date played
    :param attack: dict of teams with their respective attack scores
    :param defence: dict of teams with their respective defence scores
    :param home_adv: home advantage param
    :param zeta: decay parameter
    :return: Gives the (decayed) likelihood of a set of games assuming independence
    """
    df = df_games.copy()
    max_date = df.fixture_date.max()
    f = partial(partial_likelihood, attack=attack, defence=defence, home_adv=home_adv)
    df['calc'] = df.apply(f, axis=1)
    df['log_calc'] = df.calc.apply(np.log)
    df['decay'] = df.fixture_date.apply(lambda x: decay((max_date - df.fixture_date[0]).days, zeta=zeta))
    df['log_likelihood'] = df.decay * df.log_calc

    return df.log_likelihood.sum()


class DoublePoisson:

    def __init__(
            self,
            zeta: float = 0.002,
    ):

        self.zeta = zeta

        # TODO: Find a neater way to do this
        self.df_res = None
        self.attack = {}
        self.defence = {}
        self.home_advantage = 1

        # TODO: Do I need to store training data as part of object?
        self.df_matches = None
        self.teams = None

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: Input dataframe containing the columns fixture_date, home_team_name, away_team_name, goals_home, goals_away
        Fit the model by maximising the likelihood based on the set of games provided
        """
        required_columns = ['fixture_date', 'home_team_name', 'away_team_name', 'goals_home', 'goals_away']
        err_msg = 'Input dataframe must contain the following columns: {}'.format(','.join(required_columns))
        assert set(required_columns).issubset(data.columns), err_msg

        self.df_matches = data
        self.teams = list(set(data.home_team_name) | set(data.away_team_name))
        attack = {team: 1 for team in self.teams}
        defence = {team: 1 for team in self.teams}
        home_adv = 1.2

        x0 = np.array([x for x in chain(attack.values(), defence.values(), [home_adv])])

        # To help the model not diverge, set the attack/defence bounds to a reasonable range
        bounds = [(0.01, 10)] * (len(self.teams) * 2)
        bounds.append((1, 10))

        print('Fitting model using {} matches...'.format(len(data)), end='')
        res = minimize(self.fun, x0, bounds=bounds, method='Powell')
        print('done')
        return self.format_args(res)

    def test(self, home_team: str, away_team: str) -> MarketOdds:
        """
        :param home_team: home team name
        :param away_team: away team name
        :return: returns a MarketOdds object containing all derived match outcome probabilities
        """
        err_msg = f'{home_team} and {away_team} must not in pool of trained teams. Available teams are {self.teams}'
        assert home_team in self.teams and away_team in self.teams, err_msg
        return MarketOdds(
            home_score=(self.attack[home_team], self.defence[home_team]),
            away_score=(self.attack[away_team], self.defence[away_team]),
            home_adv=self.home_advantage,
        )

    def format_args(self, res: OptimizeResult) -> pd.DataFrame:
        """
        :param res: result of scipy.optimise.minimise
        :return: parsed results in a dataframe with attack, defence and home advantage score of each team
        """
        args = res.x
        n_teams = len(self.teams)
        attack = {t: a for t, a in zip(self.teams, args[:n_teams])}
        defence = {t: d for t, d in zip(self.teams, args[n_teams:2 * n_teams])}
        home_adv = args[-1]
        df_attack = pd.DataFrame.from_dict(attack, orient='index', columns=['attack'])
        df_defence = pd.DataFrame.from_dict(defence, orient='index', columns=['defence'])
        df_parsed = df_attack.merge(df_defence, left_index=True, right_index=True)
        df_parsed['home_adv'] = home_adv

        self.attack = attack
        self.defence = defence
        self.home_advantage = home_adv

        self.df_res = df_parsed
        return df_parsed

    def fun(self, *args) -> float:
        """
        function to be minimised. First parse the arguments and pass to the negative log likelihood
        """
        args = args[0]
        n_teams = len(self.teams)
        attack = {t: a for t, a in zip(self.teams, args[:n_teams])}
        defence = {t: d for t, d in zip(self.teams, args[n_teams:2 * n_teams])}
        home_adv = args[-1]

        return -log_likelihood(
            self.df_matches,
            zeta=self.zeta,
            attack=attack,
            defence=defence,
            home_adv=home_adv,
        )

    def __repr__(self):
        return self.df_res.to_string()


Q = r'''
select
  dlf.league_name, 
  dlf.season,
  dff.fixture_date,
  dtfh.team_name as home_team_name, 
  dtfa.team_name as away_team_name, 
  dff.teams_home_winner, 
  CASE
    WHEN dff.goals_home = dff.goals_away THEN 'DRAW' 
    WHEN dff.goals_home > dff.goals_away THEN dtfh.team_name
    WHEN dff.goals_home < dff.goals_away THEN dtfa.team_name
  END as winner,
  dff.goals_home, 
  dff.goals_away,
from dim_fixtures_fa dff
inner join dim_leagues_fa dlf
  on dlf.league_id = dff.league_id and dff.league_season = dlf.season
inner join dim_teams_fa dtfh
  on dtfh.team_id = dff.teams_home_id
inner join dim_teams_fa dtfa
  on dtfa.team_id = dff.teams_away_id
where 1=1
  and dlf.league_name IN ('{league_name}')
  and goals_home is not null and goals_away is not null
  and fixture_date >= '{fixture_date_from}'
  and fixture_date < '{fixture_date_to}'
;
'''

if __name__ == '__main__':
    fn_pkl = 'double_poisson_model.pkl'

    fixture_date_train_from = '2021-01-01'
    fixture_date_train_to = '2022-01-01'
    fixture_date_test_to = '2022-03-01'

    leagues = ['Premier League']
    if isinstance(leagues, list):
        leagues = "\',\'".join(leagues)

    query_train = Q.format(
        league_name=leagues,
        fixture_date_from=fixture_date_train_from,
        fixture_date_to=fixture_date_test_to,
    )

    df_train_test = QuestDB().execute_query(query_train)
    df_train = df_train_test[df_train_test.fixture_date <= fixture_date_train_to]
    df_test = df_train_test[df_train_test.fixture_date > fixture_date_train_to]

    dp = DoublePoisson(zeta=0.002)

    dp.fit(data=df_train)

    print(dp)

    expected_winner_list = []
    for i, row in df_test.iterrows():
        mo = dp.test(row.home_team_name, row.away_team_name)
        outcomes = [row.home_team_name, 'DRAW', row.away_team_name]
        expected_winner = outcomes[np.argmax(mo.match_odds())]
        expected_winner_list.append(expected_winner)

    df_test['expected_winner'] = expected_winner_list
    df_test['is_match'] = df_test.expected_winner == df_test.winner

    save(dp, fn_pkl)
    dp = load(fn_pkl)
    print(dp.df_res.sort_values('attack'))
