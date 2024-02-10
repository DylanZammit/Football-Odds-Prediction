import pandas as pd
from scipy.stats import skellam, poisson
from itertools import permutations
from typing import Tuple


class MarketOdds:
    """
    Provide the attack/defence score as per Dixon paper,
    and return possible pre-match probabilities
    """

    def __init__(
        self,
        home_score: Tuple[float, float],
        away_score: Tuple[float, float],
        home_adv: float,
    ):
        """
        :param home_score: tuple of attack/defence score of home team
        :param away_score:  tuple of attack/defence score of away team
        :param home_adv: home advantage parameter
        """
        self.home_param = home_score[0] * away_score[1] * home_adv
        self.away_param = away_score[0] * home_score[1]

    def match_odds(
        self,
        half: bool = False
    ) -> Tuple[float, float, float]:
        """
        :param half: is it halftime?
        :return: probability of home, draw, win
        """
        home_param = self.home_param * (1 if half else 2) / 2
        away_param = self.away_param * (1 if half else 2) / 2

        away = skellam.cdf(-0.1, home_param, away_param)
        home = 1 - skellam.cdf(0.1, home_param, away_param)
        draw = 1 - home - away
        return home, draw, away

    def score(
        self,
        home_goals: int,
        away_goals: int,
    ) -> float:
        """
        :param home_goals: goals scored by home team
        :param away_goals: goals scored by away team
        :return: probability of input result
        """
        home_goal_prob = poisson.pmf(home_goals, self.home_param)
        away_goal_prob = poisson.pmf(away_goals, self.away_param)
        return home_goal_prob * away_goal_prob

    def over_under(
        self,
        threshold: float,
        half: bool = False,
    ) -> Tuple[float, float]:
        """
        :param threshold: threshold of total number of goals scored, ex. 1.5, 2.5, 3.5 etc
        :param half: is it halftime?
        :return: probability of over/under
        """
        home_param = self.home_param * (1 if half else 2) / 2
        away_param = self.away_param * (1 if half else 2) / 2
        under = poisson.cdf(threshold, home_param + away_param)
        over = 1 - under
        return over, under

    def both_to_score(self) -> float:
        """
        :return: probability of both teams to score. Equivalent to over_under(0.5)
        """
        return (1 - poisson(0, self.home_param)) * (1 - poisson(0, self.away_param))

    def __repr__(self):
        """
        :return: Display all event probabilities of the match
        """
        match_odds = f'1x2: {self.match_odds()}'
        match_odds_ht = f'Half Time 1x2: {self.match_odds(half=True)}'

        scores = '\n'.join([f'Correct Score {h}-{a}: {self.score(h, a)}' for h, a in permutations(range(4), 2)])
        ou = '\n'.join([f'Over/Under {t+0.5}: {self.over_under(t+0.5)}' for t in range(1, 4)])

        return '\n\n'.join([match_odds, match_odds_ht, scores, ou])


if __name__ == '__main__':
    mo = MarketOdds(
        home_score=(2, 0.5),
        away_score=(2, 0.5),
        home_adv=1.2,
    )

    print(mo)
