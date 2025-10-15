import numpy as np
import dynamic_rebalancing as dr
from user import User


def run(start_date, end_date):
    # Control rebalancing frequency and amount of data to use
    rebalancing_freq = 'Q'  # 'Q' or 'M'
    rebalancing_window = 3

    # Control the regression method used and the factor model used
    regression_method = 'LASSO'  # 'LASSO', 'Ridge', 'OLS' or 'LSTM'
    factor_model = 'Carhart'  # 'CAPM', 'Fama_French_3_factors', 'Fama_French_5_factors', 'Carhart', or 'PCA'.

    regime_window = 30  # Number of days to use in regime weighted average

    # Settings
    overwrite_asset_allocation = False  # Overwrite the allocation selected by regime detection
    allow_dynamic_asset_rebalancing = True  # Allow asset allocation to switch
    regime_detection = True  # Enable regime detection to change risk level. If False, use overwritten_regime_detection
    track_return_goal = True  # Enable return goal to make changes to risk level of portfolio
    enforce_allocation = True  # F: Remove all constraints on allocation by asset class

    # Control cardinality and transaction costs
    cardinality = False
    transaction_costs = True

    # Overwritten settings
    overwritten_regime_detection = ('nothing', 2, 'MVO')  # Overwritten regime detection response. ONLY FOR EVAL USE
    overwritten_asset_allocation = {
        'alternatives': 0.,
        'bonds': 0.3,
        'commodities': 0.,
        'equity_etfs': 0.,
        'reit': 0.,
        'individual_stocks': 0.7
    }

    # Input overwritten allocation vector for bactest, will only be used if overwrite_asset_allocation = True
    b = list(overwritten_asset_allocation.values())
    b = np.array(b)

    # Create user
    person = User(risk_level=2, concentration=2, time_horizon=5, return_goal=0.6)

    # Run backtest to generate portfolio
    port = dr.run_simulation_dynamic(start_date, end_date, rebalancing_freq, rebalancing_window, person,
                                     factor_model, regression_method, min_days_between_rebalance=40,
                                     overwrite_allocation=overwrite_asset_allocation,
                                     overwritten_allocation=b,
                                     allow_dynamic_asset_rebalancing=allow_dynamic_asset_rebalancing, reduced_data=True,
                                     cardinality=cardinality, transaction_costs=transaction_costs,
                                     regime_window=regime_window, regime_detection=regime_detection,
                                     overwritten_regime_detection=overwritten_regime_detection,
                                     track_return_goal=track_return_goal, enforce_allocation=enforce_allocation)

    # Print portfolio Sharpe Ratio and Turnover
    port_SR, port_turnover = port.get_performance_statistics()
    print("SR and Turnover: {:.2f}%, {:.2f}%".format(100 * port_SR, 100 * port_turnover))

    # Print portfolio total return
    print("Total return: {:.2f}%".format(100 * (port.value[-1] / port.value[0] - 1)))

    # Plot regimes throughout backtest
    dr.plot_regimes(port.regime_dates, port.regimes)

    # Compare resuts against portfolio that is 30% BND 70% SPY
    # NOTE: BND only exists since 2008, if you want to backtest further back than that, remove this line and use
    # port.plot_value()
    dr.compare_results(port, start_date, end_date, allocation=np.array([0.3, 0.7]))
    # port.plot_value()

    # Show portfolio weights
    port.plot_weights()

    # Show portfolio risk level
    port.plot_risk_level()

    # Show portfolio asset allocation
    port.plot_asset_allocation()


def backtest_1():
    start_date = '2007-12-31'
    end_date = '2011-12-31'

    run(start_date, end_date)


def backtest_2():
    start_date = '2012-12-31'
    end_date = '2016-12-31'

    run(start_date, end_date)


if __name__ == '__main__':

    backtest_1()

    backtest_2()
