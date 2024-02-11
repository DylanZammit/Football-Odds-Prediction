from fastapi import FastAPI
from football_odds.models import load
from football_odds.utils import ROOT_DIR
import os
import uvicorn


app = FastAPI()

model = load(os.path.join(ROOT_DIR, 'API', 'model.pkl'))


@app.get("/")
def root():
    return {"message": "Hello, this is a very simple API."}


@app.get("/MATCH_ODDS/{home_team}/{away_team}")
def read_item(home_team: str, away_team: str):

    try:
        res = model.test(home_team=home_team, away_team=away_team).match_odds()
    except AssertionError:
        err_msg = f'One of {home_team}, {away_team} does not exist in model'
        return {'status': -1, 'message': err_msg}

    return {
        'home': res[0],
        'draw': res[1],
        'away': res[2],
    }


# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

    # EXAMPLE: http://127.0.0.1:8000/MATCH_ODDS/Arsenal/Leicester