import numpy as np
import time
from functions import *

# 実行不可能解の評価値
INFEASIBLE = np.finfo(float).max

# CR_FM_NES
#  d: 次元数
#  f: 目的関数
#  lam: 子個体数（偶数）
#  m: 初期平均ベクトル
#  sig: 初期ステップサイズ
#  v, D: 初期共分散行列 A = D(I + vv')D
#  stop_condition(besteval, g): 終了判定を行う関数．bestevalは最良値，gは世代数．
def CR_FM_NES(d, f, lam, m, sig, v, D, stop_condition):
    w_rank_hat = (np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))).reshape(lam, 1)
    w_rank_hat[np.where(w_rank_hat < 0)] = 0
    w_rank = w_rank_hat / sum(w_rank_hat) - ( 1 / lam )
    mu_eff = 1 / ((w_rank + (1 / lam)).T @ (w_rank + (1 / lam)))[0][0]
    c_s = (mu_eff + 2) / (d + mu_eff + 5)
    c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
    h_inv = estimateExpansionRatio(d)
    eta_m = 1
    c_1cma = 2 / ((d + 1.3) ** 2 + mu_eff)
    eta_s_move = 1
    p_s = np.zeros((d, 1))
    p_c = np.zeros((d, 1))
    eps = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
    z = np.zeros((d, lam))
    x = np.zeros((d, lam))
    idxp = np.arange(lam / 2, dtype=int)
    idxm = np.arange(lam / 2, lam, dtype=int)
    g = 0
    while True:
        # 子個体の生成
        zhalf = np.random.randn(d, lam // 2)  # / -> // TODO 1
        z[:, idxp] = zhalf
        z[:, idxm] = -zhalf
        normv = np.linalg.norm(v)
        normv2 = normv ** 2
        vbar = v / normv
        y = z + (np.sqrt(1 + normv2) - 1) * vbar @ (vbar.T @ z)
        x = m + sig * y * D
        eval, feasible = f(x)  # separate x_i to evaluate TODO 2
        sortedIndices = sortIndicesBy(eval, z) # 評価値に基づきソートされた添え字リスト (reverse it in rl task) TODO 3
        eval = eval[sortedIndices]
        z = z[:, sortedIndices]
        y = y[:, sortedIndices]
        x = x[:, sortedIndices]
        besteval = eval[0]
        p_s = (1 - c_s) * p_s + np.sqrt(c_s * (2 - c_s) * mu_eff) * (z @ w_rank)  # 進化パス
        normp_s = np.linalg.norm(p_s)
        # 学習率の計算
        eta_B = np.tanh((np.min([0.02 * np.sum(feasible), 3 * np.log(d)]) + 5 ) / (0.23 * d + 25))
        c_1 = c_1cma * (d - 5) / 6 * (np.sum(feasible) / lam)
        if normp_s >= eps:
            alpha = h_inv * min([1, np.sqrt(lam / d)]) * np.sqrt(sum(feasible) / lam)
            w_dist_hat = np.exp(alpha * np.sqrt(np.sum(z ** 2, axis=0))).reshape(lam, 1)
            a = 1 / (w_rank_hat.T @ w_dist_hat)[0][0]
            w_dist = w_rank_hat * w_dist_hat / (w_rank_hat.T @ w_dist_hat)[0][0] - ( 1 / lam )
            w = w_dist
            eta_s = eta_s_move
            l_c = 1.0
        elif normp_s >= 0.1 * eps:
            w = w_rank
            eta_s_stag = np.tanh((0.024 * np.sum(feasible) + 0.7 * d + 20) / (d + 12))
            eta_s = eta_s_stag
            l_c = 0.0
        else:
            w = w_rank
            eta_s_conv = 2 * np.tanh((0.025 * np.sum(feasible) + 0.75 * d + 10) / (d + 4))
            eta_s = eta_s_conv
            l_c = 0.0
        wxm = (x - m) @ w
        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * wxm / sig #進化パス
        m = m + eta_m * wxm #中心ベクトルの更新
        # s, tの更新
        exY = np.append(y, p_c / D, axis=1) #d x lam + 1
        yy = exY * exY #d x lam + 1
        inyvbar = vbar.T @ exY #1 x lam + 1
        yvbar = exY * vbar #d x lam+1
        gammav = 1 + normv2 #scalar
        normv4 = normv2 ** 2 #scalar
        vbarbar = vbar * vbar #d x 1
        alphavd = np.min([1, np.sqrt(normv4 + (2 * gammav - np.sqrt(gammav)) / np.max(vbarbar)) / (2 + normv2)]) #scalar
        t = exY * inyvbar - vbar * (inyvbar ** 2 + gammav ) / 2 #d x lam+1
        b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2 #scalar
        H = np.ones((d, 1)) * 2 - (b + 2 * alphavd ** 2) * vbarbar #d x 1
        invH = H ** (-1) #d x 1
        s_step1 = yy - normv2 / gammav * (yvbar * inyvbar) + np.ones((d, lam + 1)) #d x lam + 1
        invbart = vbar.T @ t #1 x lam + 1
        s_step2 = s_step1 - alphavd / gammav * ((2 + normv2) * (t * vbar) - normv2 * vbarbar @ invbart) #d x lam+1
        invHvbarbar = invH * vbarbar #d x 1
        ins_step2invHvbarbar = invHvbarbar.T @ s_step2 #1 x lam + 1
        s = (s_step2 * invH) - b / (1 + b * vbarbar.T @ invHvbarbar) * invHvbarbar @ ins_step2invHvbarbar #d x lam + 1
        insvbarbar = vbarbar.T @ s #1 x lam + 1
        t = t - alphavd * ((2 + normv2) * (s * vbar) - vbar @ insvbarbar) #d x lam+1
        # v, Dの更新
        exw = np.append(eta_B * w, np.array([l_c * c_1]).reshape(1, 1), axis=0) #lam + 1 x 1
        oldv = v
        v = v + (t @ exw) / normv
        oldD = D
        D = D + (s @ exw) * D
        # detAold, detAの計算
        nthrootdetAold = np.exp(np.sum(np.log(oldD)) / d + np.log(1 + oldv.T @ oldv) / (2 * d))[0][0]
        nthrootdetA = np.exp(np.sum(np.log(D)) / d + np.log(1 + v.T @ v) / (2 * d))[0][0]
        # s, Dの更新
        G_s = np.sum((z * z - np.ones((d, lam))) @ w) / d
        if normp_s >= eps and G_s < 0:
            l_s = 1.0
        else:
            l_s = 0.0
        sig = sig * np.exp((1 - l_s) * eta_s / 2 * G_s) * nthrootdetAold
        D = D / nthrootdetA
        g = g + 1
        print('{0} {1}'.format(g, besteval))
        if stop_condition(besteval, g):
            break

def sortIndicesBy(evals, z):
    lam = evals.size
    sortedIndices = np.argsort(evals)
    sortedEvals = evals[sortedIndices]
    noOfFeasibleSolutions = np.where(sortedEvals != INFEASIBLE)[0].size
    if noOfFeasibleSolutions != lam:
        infeasibleZ = z[:, np.where(evals == INFEASIBLE)[0]]
        distances = np.sum(infeasibleZ ** 2, axis=0)
        infeasibleIndices = sortedIndices[noOfFeasibleSolutions:]
        indicesSortedByDistance = np.argsort(distances)
        sortedIndices[noOfFeasibleSolutions:] = infeasibleIndices[indicesSortedByDistance]
    return sortedIndices

def estimateExpansionRatio(n):
    numitr = 1000
    expansion = 5
    for itr in range(numitr):
        square_expansion = expansion ** 2
        linearfitting = 0.24 * (n + 10)
        bunbo = (1 + square_expansion) * np.exp(square_expansion / 2) - linearfitting
        bunsi = (square_expansion + 3) * expansion * np.exp(square_expansion / 2)
        expansion = expansion - 0.5 * ( bunbo / bunsi )
    return expansion


if __name__ == "__main__":
    np.random.seed(100)
    d = 200
    lam = 100
    m = np.zeros((d, 1))
    sig = 2
    v = np.random.randn(d, 1) / d
    D = np.ones((d, 1))
    f = rosenbrock
    stop_condition = lambda besteval, g : besteval < 1e-7 or g > 1.0e10
    start = time.time()
    CR_FM_NES(d, f, lam, m, sig, v, D, stop_condition)
    elapsed_time = time.time() - start
    print('*** elapased_time:{0}'.format(elapsed_time) + ' [sec] ***')