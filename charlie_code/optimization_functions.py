import cvxpy as cp
import numpy as np
from cvxpy import CPLEX
from scipy.stats.distributions import chi2
from cvxpy.atoms.affine.wraps import psd_wrap
import gurobipy as gp
from gurobipy import GRB


def simple_mvo(mu, Sigma, R, C, b, log_results=0):
    n = len(mu)

    '''
    x = cp.Variable(n)

    ret = mu.T @ x
    risk = cp.quad_form(x, Sigma)

    prob = cp.Problem(cp.Minimize(risk), [cp.sum(x) == 1, x >= 0, ret >= R])
    prob.solve()
    return x
    '''

    # Model
    m = gp.Model('simple_mvo')
    m.setParam('LogToConsole', log_results)

    # Variables
    x = m.addMVar(n)

    # Expressions
    portfolio_risk = x @ Sigma @ x

    # Objective
    m.setObjective(portfolio_risk, GRB.MINIMIZE)

    # Constraints
    m.addConstr(x.sum() == 1, 'budget')
    m.addConstr(C @ x == b, 'allocation')

    # Solve
    m.optimize()

    return x.X


def constrained_mvo(mu, Sigma, lm, log_results=0):
    n = len(mu)

    # Model
    m = gp.Model('constrained_mvo')
    m.setParam('LogToConsole', log_results)

    # Variables
    x = m.addMVar(n)

    # Expressions
    risk = lm * x.T @ Sigma @ x
    ret = mu.T @ x

    # Objective
    m.setObjective(risk - ret, GRB.MINIMIZE)

    # Constraints
    m.addConstr(x.sum() == 1, 'budget')

    # Solve
    m.optimize()
    return x.X


def robust_mvo(mu, Sigma, lambda_, alpha, T, C, b, ub=1, lb=0, cardinality=False, k=10, log_results=0,
               transaction_penalty=0, x_last=None, include_trans_cost=False, k_stock=20):
    """


    :param mu:
    :param Sigma:
    :param lam:
    :param alpha:
    :param T:
    :param ub:
    :param lb:
    :param cardinality:
    :param k:
    :param log_results:
    :param transaction_penalty:
    :param x_last:
    :param include_trans_cost:

    :return:
    """
    n = len(mu)

    # Uncertainty set
    epsilon = np.sqrt(chi2.ppf(alpha, df=n))

    var = np.diag(Sigma)
    st_dev = np.sqrt(var)
    theta_half = 1 / np.sqrt(T) * np.diag(st_dev)

    # Model
    m = gp.Model('robust_mvo')
    m.setParam('LogToConsole', log_results)
    m.Params.MIPGap = 0.05
    m.Params.TimeLimit = 60

    # Variables
    x = m.addMVar(n)
    norm = m.addMVar((1,))
    diff = m.addMVar(n)

    # Expressions
    risk = lambda_ * x.T @ Sigma @ x
    ret = mu.T @ x

    # Objective
    if include_trans_cost:
        z = m.addMVar(n)
        m.setObjective(risk - ret + norm + transaction_penalty * z.sum(), GRB.MINIMIZE)
        m.addConstrs((x[i] - x_last[i] <= z[i] for i in range(n)), 'pos_aux_trans_limit')
        m.addConstrs((x_last[i] - x[i] <= z[i] for i in range(n)), 'neg_aux_trans_limit')
    else:
        m.setObjective(risk - ret + norm, GRB.MINIMIZE)

    # Constraints
    m.addConstr(x.sum() == 1, 'budget')
    m.addConstr(C @ x == b, 'allocation')

    m.addConstr(norm[0] == gp.norm(diff, 2), 'norm_const')
    m.addConstr(diff == (epsilon ** 2) * (theta_half @ x), 'diff_const')

    if cardinality:
        y = m.addMVar(n, vtype=GRB.BINARY)
        stock_indicator = C[-1, :]

        m.addConstrs((ub * y[i] >= x[i] for i in range(n)), 'ub')
        m.addConstrs((lb * y[i] <= x[i] for i in range(n)), 'lb')

        # m.addConstr(y.sum() >= k, 'cardinality')
        m.addConstr(stock_indicator @ y <= k_stock, 'stock_cardinality')
    else:
        m.addConstr(ub >= x, 'ub')
        m.addConstr(lb <= x, 'lb')

    # Solve
    m.optimize()

    return x.X


def cvar(alpha, returns, C, b, cardinality=False, ub=1, lb=0, k_stock=10, log_results=0, include_trans_cost=False,
         transaction_penalty=0.01, x_last=None):
    """

    :param mu:
    :param alpha:
    :param returns:
    :param cardinality:
    :param ub:
    :param lb:
    :param k:
    :return:
    """
    n = returns.shape[1]
    S = returns.shape[0]
    returns = np.array(returns)

    # Model
    m = gp.Model("cvar")
    m.setParam('LogToConsole', log_results)
    m.Params.MIPGap = 0.05
    m.Params.TimeLimit = 60

    # Variables
    x = m.addMVar(n)
    gamma = m.addVar()
    z = m.addMVar(S, lb=0)

    # Expressions
    losses = -1 * (returns @ x)

    # Objective
    # Transaction cost
    if include_trans_cost:
        v = m.addMVar(n)
        m.setObjective(gamma + 1 / ((1 - alpha) * S) * z.sum() + transaction_penalty * v.sum(), GRB.MINIMIZE)
        m.addConstrs((x[i] - x_last[i] <= v[i] for i in range(n)), 'pos_aux_trans_limit')
        m.addConstrs((x_last[i] - x[i] <= v[i] for i in range(n)), 'neg_aux_trans_limit')
    else:
        m.setObjective(gamma + 1 / ((1 - alpha) * S) * z.sum(), GRB.MINIMIZE)

    # Constraints
    m.addConstrs((z[i] >= losses - gamma for i in range(S)), "z_lb")
    m.addConstr(C @ x == b, 'allocation')
    m.addConstr(x.sum() == 1, 'budget')

    # Cardinality
    if cardinality:
        y = m.addMVar(n, vtype=GRB.BINARY)
        stock_indicator = C[-1, :]

        m.addConstrs((ub * y[i] >= x[i] for i in range(n)), 'ub')
        m.addConstrs((lb * y[i] <= x[i] for i in range(n)), 'lb')
        #m.addConstr(y.sum() >= k, 'cardinality')
        m.addConstr(stock_indicator @ y <= k_stock, 'stock_cardinality')

    else:
        m.addConstr(x >= lb, 'lb')
        m.addConstr(x <= ub, 'ub')

    # Solve
    m.optimize()

    return x.X


def sharpe_ratio(mu, Sigma, rf, ub=1, lb=0, log_results=0):
    """

    :param mu:
    :param Sigma:
    :param rf:
    :param ub:
    :param lb:
    :param cardinality:
    :param k:
    :return:
    """
    n = len(mu)

    rf = rf * np.ones(n)

    # Model
    m = gp.Model('rf')
    m.setParam('LogToConsole', log_results)

    # Variables
    y = m.addMVar(n)
    kappa = m.addVar(lb=0)

    # Expressions
    risk = y.T @ Sigma @ y
    excess_ret = (mu - rf).T @ y
    budget = np.ones((1, n)) @ y

    # Constraints
    m.addConstr(excess_ret == 1, 'excess_ret')
    m.addConstr(y.sum() == kappa, 'budget')
    m.addConstr(y >= lb * kappa)
    m.addConstr(y <= ub * kappa)

    # Objective

    m.setObjective(risk, GRB.MINIMIZE)

    # Solve
    m.optimize()

    x = y.X / sum(y.X)

    return x.flatten()


def robust_sharpe_ratio(mu, Sigma, rf, alpha, T, C, b, ub=1, lb=0, cardinality=False, k=50, log_results=0,
                        transaction_penalty=0, include_trans_cost=False, x_last=None, k_stock=20):
    """

    :param mu:
    :param Sigma:
    :param rf:
    :param alpha:
    :param T:
    :param cardinality:
    :param k:
    :param ub:
    :param lb:
    :return:
    """
    n = len(mu)
    epsilon = np.sqrt(chi2.ppf(alpha, df=n))

    # Uncertainty set
    var = np.diag(Sigma)
    st_dev = np.sqrt(var)
    theta_half = 1 / np.sqrt(T) * np.diag(st_dev)

    # Risk-free rate
    rf = rf * np.ones(n)

    # Model
    m = gp.Model('robust_sr')
    m.setParam('LogToConsole', log_results)

    # Variables
    y = m.addMVar(n)
    kappa = m.addVar(lb=0)
    norm = m.addMVar((1,))
    diff = m.addMVar(n)

    # Expressions
    risk = y.T @ Sigma @ y
    excess_ret = (mu - rf).T @ y

    # Constraints
    m.addConstr(excess_ret - norm >= 1, 'excess_ret')
    m.addConstr(y.sum() >= 0, 'budget')
    m.addConstr(C @ y == b * kappa, 'allocation')

    m.addConstr(norm[0] == gp.norm(diff, 2), 'norm_const')
    m.addConstr(diff == (epsilon ** 2) * (theta_half @ y), 'diff_const')

    if cardinality:
        z = m.addMVar(n, vtype=GRB.BINARY)
        v = m.addMVar(n)
        stock_indicator = C[-1, :]

        m.addConstrs((y[i] >= lb * v[i] for i in range(n)), 'y_lb')
        m.addConstrs((y[i] <= ub * v[i] for i in range(n)), 'y_ub')
        m.addConstrs((v[i] >= kappa * z[i] for i in range(n)), 'v_lb')
        m.addConstrs((v[i] <= kappa * z[i] for i in range(n)), 'v_ub')

        # m.addConstr(z.sum() >= k, 'cardinality')
        m.addConstr(stock_indicator @ z >= k_stock, 'stock_cardinality')

    else:
        m.addConstr(y >= lb * kappa, 'y_lb')
        m.addConstr(y <= ub * kappa, 'y_ub')

    # Objective
    if include_trans_cost:
        u = m.addMVar(n)
        m.setObjective(risk + transaction_penalty * u.sum(), GRB.MINIMIZE)
        m.addConstrs((y[i] <= kappa * (u[i] + x_last[i]) for i in range(n)), 'pos_aux_trans_limit')
        m.addConstrs((-y[i] <= kappa * (u[i] - x_last[i]) for i in range(n)), 'neg_aux_trans_limit')
    else:
        m.setObjective(risk, GRB.MINIMIZE)

    # Solve
    m.optimize()

    x = y.X / sum(y.X)

    return x.flatten()


def risk_parity(Sigma, C, b, log_results=0):
    """

    :param mu:
    :param Sigma:
    :return:
    """

    """
    n = Sigma.shape[0]
    
    # Variables
    c = np.ones((n,1))
    y = cp.Variable((n, 1))

    # Expressions
    risk = cp.quad_form(y, psd_wrap(Sigma))

    # TODO implement allocation constraints
    # Constraints
    constraints = [y >= 0]

    # Problem
    prob = cp.Problem(cp.Minimize(0.5 * risk - c.T @ cp.log(y)), constraints)
    prob.solve(solver=CPLEX)

    x = y.value / sum(y.value)

    return x.flatten()
    """
    n = Sigma.shape[0]

    # Model
    m = gp.Model('risk_parity')
    m.setParam('LogToConsole', log_results)

    # Variables
    x = m.addMVar(n)
    theta = m.addVar()
    z = m.addMVar(n)
    v = m.addMVar(n)

    # Expressions
    risk = Sigma @ x

    # obj = gp.quicksum(((x[i] * risk[i]) - theta)**2 for i in range(n))

    # Objective
    m.setObjective(z.sum(), GRB.MINIMIZE)

    # Constraints
    m.addConstr(x.sum() == 1, 'budget')
    m.addConstr(C @ x == b, 'allocation')

    m.addConstrs((x[i] * x[i] * v[i] * v[i] - 2 * x[i] * v[i] * theta + theta * theta >= z[i] for i in range(n)), 'lim')
    m.addConstrs((v[i] == gp.quicksum(Sigma[i, j] * x[j] for j in range(n)) for i in range(n)), 'v constraint')

    # Solve
    m.optimize()

    return x.X


def minimum_variance(Sigma, C, b, ub=1, lb=0, cardinality=False, k=10, log_results=0,
                     transaction_penalty=0, x_last=None, include_trans_cost=False, k_stock=20):
    """


       :param mu:
       :param Sigma:
       :param lam:
       :param alpha:
       :param T:
       :param ub:
       :param lb:
       :param cardinality:
       :param k:
       :param log_results:
       :param transaction_penalty:
       :param x_last:
       :param include_trans_cost:

       :return:
       """
    n = Sigma.shape[0]

    # Model
    m = gp.Model('robust_mvo')
    m.setParam('LogToConsole', log_results)
    m.Params.MIPGap = 0.05
    m.Params.TimeLimit = 60

    # Variables
    x = m.addMVar(n)

    # Expressions
    risk = x.T @ Sigma @ x

    # Objective
    if include_trans_cost:
        z = m.addMVar(n)
        m.setObjective(risk + transaction_penalty * z.sum(), GRB.MINIMIZE)
        m.addConstrs((x[i] - x_last[i] <= z[i] for i in range(n)), 'pos_aux_trans_limit')
        m.addConstrs((x_last[i] - x[i] <= z[i] for i in range(n)), 'neg_aux_trans_limit')
    else:
        m.setObjective(risk, GRB.MINIMIZE)

    # Constraints
    m.addConstr(x.sum() == 1, 'budget')
    m.addConstr(C @ x == b, 'allocation')

    if cardinality:
        y = m.addMVar(n, vtype=GRB.BINARY)
        stock_indicator = C[-1, :]

        m.addConstrs((ub * y[i] >= x[i] for i in range(n)), 'ub')
        m.addConstrs((lb * y[i] <= x[i] for i in range(n)), 'lb')

        # m.addConstr(y.sum() >= k, 'cardinality')
        m.addConstr(stock_indicator @ y == k_stock, 'stock_cardinality')
    else:
        m.addConstr(ub >= x, 'ub')
        m.addConstr(lb <= x, 'lb')

    # Solve
    m.optimize()

    return x.X
