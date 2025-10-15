import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import scipy.stats

matplotlib.use('TkAgg')


class Portfolio:
    """
    We need a portfolio class to keep track how our portfolio is doing. This class achieves this goal.
    """

    def __init__(self, tickers, initial_weights, starting_time, initial_allocation, initial_value=1, name=None):
        """
        Portfolio Class Initialization.
        """
        self.num_assets = len(tickers)
        self.tickers = tickers
        self.weights = initial_weights.reshape((1, self.num_assets))
        self.value = np.array([initial_value])
        self.returns = np.zeros((1,self.num_assets))
        self.asset_allocation = initial_allocation.reshape((1, 6))
        self.total_returns = np.array([0])
        self.time = [starting_time]
        self.name = name
        self.risk_level = []
        self.optimization_method = []
        self.turnover = [0]
        self.regime_dates = None
        self.regimes = None

    def add_new_weights(self, new_weights):
        """
        This method adds new weights when a new period is on.
        """
        self.weights = np.vstack((self.weights, new_weights))


    def add_period_returns(self, new_returns):
        """
        This method add new returns when new return is on.
        """
        self.returns = np.vstack((self.returns, new_returns))

    def add_cumulative_portfolio_return(self, new_return):
        """
        This method adds new total portfolio returns and calculates the portfolio value for the period.
        """
        self.total_returns = np.append(self.total_returns, new_return)

        value = self.value[-1] * (new_return + 1)
        self.value = np.append(self.value, value)

    def update_time(self, time_):
        """
        This method update the time we need to rebalance the portfolio.
        """
        self.time.append(time_)

    def calculate_turnover(self, new_weights):
        """
        Calculate portfolio turnover and add to portfolio
        :param new_weights: new x vector
        """
        turnover = np.sum(np.abs(self.weights[-1, :] - new_weights))
        self.turnover.append(turnover)

    def plot_value(self):
        """
        Plot the cumulative returns for the portfolio
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.time, self.value)
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Portfolio's Cumulative Return")
        plt.show()

    def plot_weights(self):
        """
        Plot the changing weights for the portfolio
        """
        plt.figure(figsize=(12, 6))
        plt.stackplot(self.time, self.weights.T)
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xlabel("Time")
        plt.ylabel("Individual asset weights")
        plt.show()

    def get_portfolio_snapshot(self, time):
        """
        Get weights of portfolio at certain time

        :param time:
        :return: weights at time t
        """
        try:
            idx = self.time.index(time)

        except ValueError:
            return None

        weights = self.weights[idx]

        return weights

    def plot_method(self):
        """
        plot the optimization method used

        """

        method = [0 if i=='CVaR' else 1 for i in self.optimization_method]

        plt.figure(figsize=(12, 6))
        plt.plot(self.time, method)
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Optimization method (1 - MVO, 0 - CVaR")
        plt.show()

    def plot_risk_level(self):
        """
        Plot the portfolio risk level over time

        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.time, self.risk_level)
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Risk level")
        plt.show()

    def plot_asset_allocation(self):
        """
        plot the portfolio asset class allocation over time

        """
        plt.figure(figsize=(12, 6))
        plt.stackplot(self.time, self.asset_allocation.T)
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xlabel("Time")
        plt.ylabel("Asset class weights")
        plt.legend(['Alternatives', 'Bonds', 'Commodities', 'Equity ETFs', 'REITs', 'Individual stocks'])
        plt.show()

    def get_performance_statistics(self):
        """
        Return the SR and avergae turnover for the portfolio throughout the tested period
        :return: SR, turnover
        """
        SR = (scipy.stats.gmean(self.total_returns + 1) - 1) / np.std(self.total_returns)
        avg_turnover = np.mean(self.turnover)

        return SR, avg_turnover