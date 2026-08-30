from cvxpy.error import SolverError
import numpy as np
import cvxpy as cp
import traceback

class MeanVarOptimizer:

    def __init__(self, n_assets, opti_kwargs):
        self.pos_scale = 1e4
        self.vola_scale = 1e4

        self.opti_kwargs = opti_kwargs

        self.n_assets = n_assets

        self.max_book = cp.Constant(opti_kwargs['max_book'] / self.pos_scale)
        self.risk_penalty_coef = cp.Constant(opti_kwargs['risk_penalty_coef'])

        self._gen_prob()

    def _gen_prob(self):
        # 单位RMB
        self.open_long = cp.Variable(self.n_assets,nonneg=True)
        self.close_long_yest = cp.Variable(self.n_assets,nonneg=True)

        self.pos_long = cp.Parameter(self.n_assets,nonneg=True)
        self.pos_long_yest = cp.Parameter(self.n_assets,nonneg=True)

        self.cost_open = cp.Parameter(self.n_assets,nonneg=True)
        self.cost_close = cp.Parameter(self.n_assets,nonneg=True)

        self.cash = cp.Parameter(1,nonneg=True)

        self.mkt_cap_limit = cp.Parameter(self.n_assets,nonneg=True)
        self.amt_limit = cp.Parameter(self.n_assets,nonneg=True)

        self.raw_fcst = cp.Parameter(self.n_assets)

        self.pos_new = self.pos_long + self.pos_long_yest + self.open_long - self.close_long_yest
        self.delta_pos = self.open_long - self.close_long_yest

        self.cost = self.open_long @ self.cost_open + self.close_long_yest @ self.cost_close

        objective = self.raw_fcst @ self.delta_pos - self.cost

        self.ret_var = cp.Parameter(self.n_assets,nonneg=True)
        self.vola_exist = cp.Parameter(self.n_assets,nonneg=True)
        # (w+Δw).T @ A @ (w+Δw) = w.T @ A @ w + Δw.T @ A @ Δw + 2 * w.T @ A @ Δw
        self.risk_penalty = cp.square(self.delta_pos) @ self.ret_var
        self.risk_penalty += 2 * self.vola_exist @ self.delta_pos

        objective -= self.risk_penalty * self.risk_penalty_coef

        contraint = [
            cp.sum(self.pos_new) <= self.max_book,
            self.pos_new <= self.mkt_cap_limit,
            self.open_long + self.close_long_yest <= self.amt_limit,
            self.close_long_yest <= self.pos_long_yest,
            cp.sum(self.open_long) + self.cost  <= (self.cash + cp.sum(self.close_long_yest)) * 0.85 # 防止出现现金不足的情况
        ]

        self.prob = cp.Problem(cp.Maximize(objective), contraint)

    def get_results(self):
        return {
                "open_long":self.open_long.value * self.pos_scale,
                "close_long_yest":self.close_long_yest.value * self.pos_scale,
        }

    def get_default_results(self):
        return {
                "open_long":np.zeros((self.n_assets,)),
                "close_long_yest":np.zeros((self.n_assets,)),
        }


    def update_param(self, **param_kwargs):

        self.ret_var.value = param_kwargs['ret_var'] / self.vola_scale

        self.cash.value = np.array([param_kwargs['cash']]) / self.pos_scale

        self.pos_long.value = param_kwargs['pos_long'] / self.pos_scale
        self.pos_long_yest.value = param_kwargs['pos_long_yest'] / self.pos_scale

        self.cost_open.value = param_kwargs['cost_open']
        self.cost_close.value = param_kwargs['cost_close']

        self.mkt_cap_limit.value = param_kwargs['mkt_cap_limit'] / self.pos_scale
        self.amt_limit.value = param_kwargs['amt_limit'] / self.pos_scale

        self.raw_fcst.value = param_kwargs['raw_fcst']

        total_long = (param_kwargs['pos_long'] + param_kwargs['pos_long_yest']) / self.pos_scale
        self.vola_exist.value = total_long * param_kwargs['ret_var'] / self.vola_scale


    def solve(self, **param_kwargs):
        self.update_param(**param_kwargs)
        # 设置MOSEK整数规划参数
        mosek_params = {}
        # {
        #     'MSK_DPAR_MIO_TOL_ABS_GAP': 1000 / self.pos_scale,  # 绝对差异设为100元
        #     "MSK_DPAR_MIO_TOL_REL_GAP": 1e-2, # 最优相对差异
        # }
        try:
            self.prob.solve(solver=cp.MOSEK,warm_start=True, mosek_params=mosek_params)
            if self.prob.status=='optimal':
                return self.get_results()
            else:
                print(f"{param_kwargs['date']}: {self.prob.status}")
                return self.get_default_results()
        except SolverError:
            try:
                self.prob.solve(solver=cp.MOSEK,warm_start=True,verbose=True, mosek_params=mosek_params)
                if self.prob.status=='optimal':
                    return self.get_results()
                else:
                    print(f"{param_kwargs['date']}: {self.prob.status}")
                    return self.get_default_results()
            except SolverError:
                traceback.print_exc()
                return self.get_default_results()

if __name__=="__main__":
    opti_kwargs = {
        "init_cash":1e8,
        "max_book":1e8 * 0.9,
        "risk_penalty_coef":1,
        "mkt_cap_limit_ratio": 0.01, # 个股持仓不超过市值的1%
        "amount_limit_ratio": 0.01, # 个股调仓不超过个股交易额的1%
        "trade_aversion_coef": 1,
        "cost_open": 3e-4 + 5e-4, # 3 bps券商佣金 + 滑点
        "cost_close": 3e-4 + 5e-4 + 5e-4, # 3 bps券商佣金 + 5 bps 印花税 + 滑点
    }

    solver = MeanVarOptimizer(5504, opti_kwargs)

    print(solver.prob.is_dpp())
