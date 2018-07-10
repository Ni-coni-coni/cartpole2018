import numpy as np
import copy

# CR_FM_NES
#  d: 次元数
#  f: 目的関数
#  lam: 子個体数（偶数）
#  m: 初期平均ベクトル
#  sig: 初期ステップサイズ
#  v, D: 初期共分散行列 A = D(I + vv')D
#  stop_condition(besteval, g): 終了判定を行う関数．bestevalは最良値，gは世代数．


class HillClimbing(object):
    def __init__(self, d, x_init, alpha=0.1):
        self.d = d
        self.x_init = x_init
        self.alpha = alpha

    def run(self, eval_func, stop_condition):
        g = 0
        best_eval = 0
        x = copy.deepcopy(self.x_init)
        while True:
            x_new = x + self.alpha * (2 * np.random.rand() - 1)
            eval = eval_func(x_new)


