"""
This script implements the Double-Poisson Model of
Dixon - Modelling Association Football Scores and Inefficiencies in the Football Betting Market
"""

from football_odds.utils import expon, decay, save, load
from scipy.optimize import minimize
from football_odds.utils.connectors import QuestDB
from typing import Union, List
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
) -> float:
    """
    :param df_games: DataFrame of fixtures containing the home/away teams, result, date played
    :param attack: dict of teams with their respective attack scores
    :param defence: dict of teams with their respective defence scores
    :param home_adv: home advantage param
    :return: Gives the (decayed) likelihood of a set of games assuming independence
    """
    df = df_games.copy()
    max_date = df.fixture_date.max()
    f = partial(partial_likelihood, attack=attack, defence=defence, home_adv=home_adv)
    df['calc'] = df.apply(f, axis=1)
    df['log_calc'] = df.calc.apply(np.log)
    df['decay'] = df.fixture_date.apply(lambda x: decay((max_date - df.fixture_date[0]).days))
    df['log_likelihood'] = df.decay * df.log_calc

    return df.log_likelihood.sum()


class DoublePoisson:
    Q = r'''
    select
      dlf.league_name, 
      dlf.season,
      dff.fixture_date,
      dtfh.team_name as home_team_name, 
      dtfa.team_name as away_team_name, 
      dff.teams_home_winner, 
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
      and season >= {season_from}
      and season <= {season_to}
    ;
    '''

    def __init__(
            self,
            leagues: Union[str, List[str]],
            season_from: int,
            season_to: int,
    ):
        """
        :param leagues: Name of the league, or list of leagues to read fixtures from
        :param season_from: load results from this season
        :param season_to: load results until this season
        """
        if isinstance(leagues, list):
            leagues = "\',\'".join(leagues)

        query_results = self.Q.format(
            league_name=leagues,
            season_from=season_from,
            season_to=season_to,
        )

        self.df_matches = QuestDB().execute_query(query_results)
        self.teams = list(set(self.df_matches.home_team_name) | set(self.df_matches.away_team_name))
        self.df_res = None

    def fit(self):
        """
        Fit the model by maximising the likelihood based on the set of games provided
        """
        attack = {team: 1 for team in self.teams}
        defence = {team: 1 for team in self.teams}
        home_adv = 1.2

        x0 = np.array([x for x in chain(attack.values(), defence.values(), [home_adv])])

        # To help the model not diverge, I set the attack/defence bounds to a reasonable range
        bounds = [(0.01, 10)] * (len(self.teams) * 2)
        bounds.append((1, 10))

        print('Fitting model...', end='')
        res = minimize(self.fun, x0, bounds=bounds, method='Powell')
        print('done')
        self.format_args(res.x)

    def format_args(self, args) -> pd.DataFrame:
        """
        :param args: result of scipy.optimise.minimise
        :return: parsed results in a dataframe with attack, defence and home advantage score of each team
        """
        n_teams = len(self.teams)
        attack = {t: a for t, a in zip(self.teams, args[:n_teams])}
        defence = {t: d for t, d in zip(self.teams, args[n_teams:2 * n_teams])}
        home_adv = args[-1]
        df_attack = pd.DataFrame.from_dict(attack, orient='index', columns=['attack'])
        df_defence = pd.DataFrame.from_dict(defence, orient='index', columns=['defence'])
        df_parsed = df_attack.merge(df_defence, left_index=True, right_index=True)
        df_parsed['home_adv'] = home_adv
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

        return -log_likelihood(self.df_matches, attack=attack, defence=defence, home_adv=home_adv)


def main():
    fn_pkl = 'double_poisson_model.pkl'
    dp = DoublePoisson(leagues='Premier League', season_from=2021, season_to=2021)

    dp.fit()
    save(dp, fn_pkl)
    dp = load(fn_pkl)
    print(dp.df_res.sort_values('attack'))


if __name__ == '__main__':
    main()
