COLLECTION = 'stocks'           # MongoDB collection
API = 'qY6NzJTzXwLqfRxHzYrv'    # Quandl API key
TABLE = 'WIKI/'                 # Quandl table
COLLAPSE = 'monthly'            # Quandl data aggregation value

import uuid
import quandl
quandl.ApiConfig.api_key = API


class Stock(object):
    def __init__(self, ticker, returns, mu, std, _id = None):
        # Stock class creates stock instances of assets stored/allowed
        # Only needs to enter ticker name and run get_Params to fill in the rest.
        self.ticker = ticker
        self.returns = returns
        self.mu = mu
        self.std = std
        self._id = uuid.uuid4().hex if _id is None else _id

    def __repr__(self):
        return "<Asset: {}>".format(self.ticker)

    @classmethod
    def get_Params(cls, ticker, start_date, end_date):
        '''
        Gets ticker data from Quandl API and saves stock to database

        :param ticker: {type:string} Asset Ticker (ex: 'AAPL')
        :param start_date: {type:string} time-series start date (ex: YYYY-MM-DD '2006-01-01')
        :param end_date: {type:string} time-series end date (ex: YYYY-MM-DD '2006-01-01')
        :return: Stock instance
        '''

        error = False
        try:
            # sets path to eleventh column (adjusted closing) of WIKI EOD table/ ticker
            code = TABLE + ticker + '.11'
            # retrieve data from Quandl API [start_date, end_date] aggregated monthly
            data = quandl.get(code, start_date=start_date, end_date=end_date, collapse=COLLAPSE)
        except quandl.errors.quandl_error.NotFoundError:
            error = True

        #TODO Change this
        if error is True:
            print("Failure")

        rets = data.pct_change().dropna()   # create return timeseries
        rets.columns = [ticker]

        # TODO change how mu is calculated. Implement factor model.
        mu = rets.mean().values[0]
        std = rets.std().values[0]

        stock = cls(ticker = ticker, returns = rets.to_json(orient='index'), mu = mu, std = std)    # create instance of stock

        return stock

    def json(self):     # Creates JSON representation of stock instance
        return{
            "_id" : self._id,
            "ticker" : self.ticker,
            "returns" : self.returns,
            "mu" : self.mu,
            "std": self.std
        }

