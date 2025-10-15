import pandas as pd
import numpy as np
import optimization_functions as of
import data_processing as dp
from user import User
from portfolio import Portfolio
from archived import pca


def get_rebalancing_increments(start_date, end_date, rebalancing_freq):
    """
    Given start and end date, and rebalancing frequency, create array of rebalancing times

    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :param rebalancing_freq: rebalancing frequency (M, or Q)
    :return: array of period times
    """
    t_start = pd.to_datetime(start_date, format='%Y-%m-%d')
    t_end = pd.to_datetime(end_date, format='%Y-%m-%d')

    if rebalancing_freq == 'Q':
        times = pd.date_range(start=t_start, end=t_end, freq='BQ', inclusive='both')

        if t_start not in times:
            times = times.insert(0, t_start)

        times_list = times.strftime('%Y-%m-%d').tolist()

        return times_list

    return None


def get_training_period(training_end_date, training_periods, rebalancing_freq):
    """
    Given start date of testing, generate x training periods before this date based on the rebalancing frequency to
    enable model to configure

    :param training_end_date: 'YYYY-MM-DD'
    :param training_periods: 'YYYY-MM-DD'
    :param rebalancing_freq: rebalancing frequency (M, or Q)
    :return: array of period times
    """
    t_end_date = pd.to_datetime(training_end_date, format='%Y-%m-%d')

    if rebalancing_freq == 'Q':
        t_start_date = t_end_date - pd.tseries.offsets.BQuarterEnd(n=training_periods)
        times = pd.date_range(start=t_start_date, end=t_end_date, freq='BQ', inclusive='both')

        if t_start_date not in times:
            times = times.insert(0, t_start_date)

        times_list = times.strftime('%Y-%m-%d').tolist()
        t_start_date = t_start_date.strftime('%Y-%m-%d')

        return t_start_date, times_list

    return None


def run_simulation(start_date, end_date, rebalancing_freq, rebalancing_window, person, factor_model):
    """
    Run simulation model between the dates specified at the rebalancing intervals and rebalacing windows specified

    :param user: instance of user used to get user preferences used in optimization
    :param rebalancing_window: number of periods to use in estimation of mu and Q in portfolio optimization
    :return: portfolio instance
    """

    # Get the time array to of periods to rebalance at
    times = get_rebalancing_increments(start_date, end_date, rebalancing_freq)

    # Get the array of training times before the start date
    training_start, training_times = get_training_period(start_date, rebalancing_window, rebalancing_freq)

    # Set the start date to the end of the training period
    training_end = start_date
    print(training_times)

    print("Training start : Training end : Test start : Test end")
    print(training_start, training_end, start_date, end_date)

    # TODO double check this in the future
    # Create array of all times to be used in optimization
    all_times = training_times + times[1:]
    print(all_times)

    # Load all data for training and test
    data = dp.load_data(training_start, end_date)
    asset_count = dp.get_asset_count(data[2])

    # TODO check if k, lb and ub are a feasible combination. Check this in user.py before returning user
    person.verify_parameters(asset_count)

    # Training data and initial weights
    training_data = dp.get_period_data(data, training_start, training_end)
    print(training_start, training_end)

    # TODO Find another way to get optimization method, move down?
    initial_weights = optimize('MVO', training_data, person, factor_model, transaction_cost=False)

    # create portfolio
    tickers = list(data[0].columns.values)
    port = Portfolio(tickers, initial_weights, start_date, initial_value=1000, name='Test Portfolio')

    # TODO add ability to manually step through on website and plot portfolio evolution and allocation
    # Run through testing period and rebalance at the frequency decided above
    for i in range(rebalancing_window, len(all_times)-1):
        period_start_time = all_times[i]
        period_data_start_time = all_times[i - rebalancing_window + 1]
        period_end_time = all_times[i + 1]

        print("Training start : Period start : Period end")
        print(period_data_start_time, period_start_time, period_end_time)

        # Get relevant data
        training_data = dp.get_period_data(data, period_data_start_time, period_start_time)
        period_data = dp.get_period_data(data, period_start_time, period_end_time)
        # Get last iterations weights to use in transactions costs
        x_last = port.weights[-1,:]

        # TODO add functionality to increase aggressiveness given returns against return goal
        # Optimize for x
        x = optimize('MVO', training_data, person, factor_model, x_last=x_last, transaction_cost=False)

        # Get actual asset returns for the period
        period_returns = get_period_returns(period_data, period_start_time, period_end_time)

        # Calculate weighted portfolio return
        portfolio_return = np.matmul(period_returns.T, x)

        # Update portfolio
        port.calculate_turnover(x)
        port.add_new_weights(x)
        port.add_period_returns(period_returns)
        port.add_cumulative_portfolio_return(portfolio_return)
        port.update_time(period_end_time)

    return port


def optimize(method, data, person, factor_model, transaction_cost=True, x_last=None):
    """
    Optimize the portfolio given the data and user's preferences, return weights x

    :param method:
    :param data: (price data, share_data, asset_data) to be used in optimization
    :param person: instance of user
    :param factor_model: selection of which factor model to use in parameter estimation
    :return: array of weights x
    """

    # Load data
    price_data, share_data, asset_data = data

    # Load user
    params = person.parameters
    allocation = person.asset_allocation

    # Get returns
    returns = dp.get_returns(price_data)
    assets = dp.get_asset_count(asset_data)
    # Construct C matrix
    C = dp.construct_allocation_matrix(assets)

    # Construct b vector
    b = list(allocation.values())
    b = np.array(b)

    # Get mu and Sigma
    # TODO enable use of factor model or PCA here
    if factor_model == 'PCA':
        mu, Sigma = pca.pca(returns, 5)
    else:
        mu = dp.get_exp_rets(returns)
        Sigma = dp.get_Q(returns)

    tickers = list(returns.columns.values)

    # Get parameters
    T = returns.shape[0]
    n = returns.shape[1]
    lambda_ = params['lambda']
    alpha = params['alpha']
    lb = params['lb']
    ub = params['ub']
    k_fraction = params['k_fraction']
    k = round(k_fraction * len(tickers))

    # Optimize
    # TODO functionality to choose solver
    # TODO maybe optimize using several solvers and select one with best sharpe ratio?
    # TODO Use decision making file?
    if method == 'MVO':
        x = of.robust_mvo(mu, Sigma, lambda_, alpha, T, C, b, x_last=x_last, lb=lb, ub=ub, cardinality=True, k=k,
                          transaction_penalty=0, include_trans_cost=transaction_cost)


    return x


def get_period_returns(data, start_date, end_date):
    """
    Return the returns for a period given the price table and start and end dates
    :param data: (price_data, share_data, asset_data) tuple
    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :return: ndarray (n x 1)
    """
    price_data = data[0]

    start_price = price_data.iloc[price_data.index.get_indexer([start_date], method='pad')[0]]

    end_price = price_data.iloc[price_data.index.get_indexer([end_date], method='pad')[0]]

    start_price = start_price.to_numpy()
    end_price = end_price.to_numpy()

    period_returns = np.divide(end_price, start_price) - 1

    return period_returns


def select_optimal_model(allowed_methods, data, person, mu, Sigma, C, b, x_last):
    """
    Use window of past returns on new weights to see optimal weights, use method

    :return:
    """
    price_data, share_data, asset_data = data
    returns = dp.get_returns(price_data)

    tickers = list(returns.columns.values)

    params = person.parameters

    T = returns.shape[0]
    n = returns.shape[1]
    lambda_ = params['lambda']
    alpha = params['alpha']
    lb = params['lb']
    ub = params['ub']
    k_fraction = params['k_fraction']
    k = round(k_fraction * len(tickers))

    rf=0

    allocations = {}
    # TODO change to regime detection
    for om in allowed_methods:
        if om == 'MVO':
            x = of.robust_mvo(mu, Sigma, lambda_, alpha, T, C, b, x_last=x_last, lb=lb, ub=ub, cardinality=True, k=k,
                              transaction_penalty=0.001, include_trans_cost=True)
            allocations[om] = x

        elif om == 'SR':
            x = of.robust_sharpe_ratio(mu, Sigma, rf, alpha, T, C, b, lb=lb, ub=ub, cardinality=True, k=k)
            allocations[om] = x

        elif om == 'CVaR':
            x = of.cvar(alpha, rf, C, b, cardinality=True, ub=1, lb=0, k=10, log_results=0, include_trans_cost=True,
                        transaction_penalty=0.001, x_last=x_last)
            allocations[om] = x

        elif om == 'RP':
            x = of.risk_parity(Sigma)
            allocations[om] = x

        elif om == 'MinVar':
            x = of.minimum_variance(Sigma, C, b, cardinality=True, ub=1, lb=0, k=10)
            allocations[om] = x

    # TODO test using past data and select with highest SR


if __name__ == '__main__':
    # parameters, move out
    start_date = '2007-12-31'
    end_date = '2011-12-31'
    rebalancing_freq = 'Q'
    rebalancing_window = 2
    factor_model=None

    person = User(aggressiveness=3, risk_tolerance=2, concentration=2, time_horizon=5)

    port = run_simulation(start_date, end_date, rebalancing_freq, rebalancing_window, person, factor_model)

    print(port.total_returns)
    print(port.turnover)
    print(port.risk_level)
    print(port.optimization_method)

    print(port.get_performance_statistics())

    port.plot_value()
    port.plot_weights()
    port.plot_method()
    port.plot_risk_level()




