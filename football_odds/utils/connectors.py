import psycopg as pg
import pandas as pd
import sqlparse
import datetime


class QuestDB:

    def __init__(
            self,
            host: str = None,
            dbname: str = None,
            port: int = None,
            user: str = None,
            password: str = None,
    ):

        # This should not be hardcoded.
        # Set up a config file
        credentials = {
            'host': '127.0.0.1',
            'dbname': 'qdb',
            'port': 8812,
            'user': 'admin',
            'password': 'quest',
        }

        host = credentials['host'] if host is None else host
        dbname = credentials['dbname'] if dbname is None else dbname
        port = credentials['port'] if port is None else port
        user = credentials['user'] if user is None else user
        password = credentials['password'] if password is None else password

        self.creds = {
            'host': host,
            'dbname': dbname,
            'port': int(port),
            'password': password,
            'user': user,
        }

    def connection(self):
        return pg.connect(**self.creds, autocommit=True)

    def execute_query(self, query, log_query=True, *args):
        datetime_object_start = str(datetime.datetime.now())
        log = print if log_query else lambda x: None

        log(f'**[{datetime_object_start}]')
        with pg.connect(**self.creds) as con:
            con.autocommit = True
            query = sqlparse.format(query, strip_comments=True).strip()
            sub_queries = sqlparse.split(query)

            for i, subquery in enumerate(sub_queries):
                log(subquery)

                with con.cursor() as cur:
                    res = cur.execute(subquery, *args)

                    try:
                        res = cur.fetchall()
                        cols = [x[0] for x in cur.description]
                        out = pd.DataFrame(res, columns=cols)
                    except Exception:  # specify error given when there is no query output
                        out = res

        datetime_object_start = str(datetime.datetime.now())
        log(f'**[{datetime_object_start}] Ended executing Query')

        return out

