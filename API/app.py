from fastapi import FastAPI
from pydantic import BaseModel
from football_odds.models import load, DoublePoisson
from football_odds.utils import ROOT_DIR
import os
import uvicorn
from enum import Enum


app = FastAPI()

model = load(os.path.join(ROOT_DIR, 'API', 'model.pkl'))


class ResultFormats(Enum):
    odds = 'odds'
    prob = 'prob'


class HomeAwayOdds(BaseModel):
    home_team: str
    away_team: str
    res_fmt: ResultFormats = 'odds'


@app.get("/")
def root():
    return {"message": "Hello, this is a very simple API."}


@app.post("/match_odds")
def match_odds(home_away_odds: HomeAwayOdds):

    try:
        res = model.get_match_odds(
            home_team=home_away_odds.home_team,
            away_team=home_away_odds.away_team
        ).match_odds()
    except AssertionError:
        err_msg = f'One of {home_away_odds.home_team}, {home_away_odds.away_team} does not exist in model'
        return {'status': -1, 'message': err_msg}

    if home_away_odds.res_fmt == ResultFormats.odds:
        return {
            'home': 1 / res[0],
            'draw': 1 / res[1],
            'away': 1 / res[2],
        }
    elif home_away_odds.res_fmt == ResultFormats.prob:
        return {
            'home': res[0],
            'draw': res[1],
            'away': res[2],
        }
    return {}


# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
