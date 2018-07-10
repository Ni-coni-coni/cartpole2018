import numpy as np


class HillClimbing(object):
    def __init__(self, d, x_init, alpha=0.1):
        self.d = d
        self.x_init = x_init
        self.alpha = alpha
        self.rs = np.random.RandomState()

    def set_seed(self, seed):
        self.rs.seed(seed)

    def run(self, eval_func, stop_condition):
        d = self.d
        x = self.x_init
        alpha = self.alpha
        rs = self.rs

        data = []
        g = 0
        best_eval = 0

        while True:
            x_new = x + alpha * (2 * rs.rand(d, 1).astype('float32') - 1)
            eval = eval_func(x_new)
            if eval > best_eval:
                best_eval = eval
                x = x_new
            g += 1
            print('g:{} best_eval:{}'.format(g, best_eval))
            data.append([g, best_eval])
            alg_stat = stop_condition(best_eval, g)
            if alg_stat == 1:
                is_success = True
                break
            elif alg_stat == -1:
                is_success = False
                break

        return is_success, data


class CR_FM_NES(object):

    def __init__(self, d, lam, m_init, sig, v, D):
        self.d = d
        self.lam = lam
        self.m_init = m_init
        self.sig = sig
        self.v = v
        self.D = D
        self.rs = np.random.RandomState()

    def set_seed(self, seed):
        self.rs.seed(seed)

    def run(self, eval_func, stop_condition):
        d = self.d
        lam = self.lam
        m = self.m_init
        sig = self.sig
        v = self.v
        D = self.D
        rs = self.rs

        w_rank_hat = (np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))).reshape(lam, 1)
        w_rank_hat[np.where(w_rank_hat < 0)] = 0
        w_rank = w_rank_hat / sum(w_rank_hat) - (1 / lam)
        mu_eff = 1 / ((w_rank + (1 / lam)).T @ (w_rank + (1 / lam)))[0][0]
        c_s = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        h_inv = self._estimate_expansion_ratio()
        eta_m = 1
        c_1cma = 2 / ((d + 1.3) ** 2 + mu_eff)
        eta_s_move = 1
        p_s = np.zeros((d, 1), dtype='float32')
        p_c = np.zeros((d, 1), dtype='float32')
        eps = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        z = np.zeros((d, lam), dtype='float32')
        x = np.zeros((d, lam), dtype='float32')
        idxp = np.arange(lam // 2, dtype=int)
        idxm = np.arange(lam // 2, lam, dtype=int)

        data = []
        g = 0

        while True:
            # 子個体の生成
            zhalf = rs.randn(d, lam//2)
            z[:, idxp] = zhalf
            z[:, idxm] = -zhalf
            normv = np.linalg.norm(v)
            normv2 = normv ** 2
            vbar = v / normv
            y = z + (np.sqrt(1 + normv2) - 1) * vbar @ (vbar.T @ z)
            x = (m + sig * y * D).astype('float32')
            x_list = np.hsplit(x, np.arange(lam)[1:])
            evals = []
            for idx in range(lam):
                evals.append(eval_func(x_list[idx]))
            sortedIndices = np.argsort(evals)[::-1]  # 評価値に基づきソートされた添え字リスト
            evals = np.array(evals)
            evals = evals[sortedIndices]
            z = z[:, sortedIndices]
            y = y[:, sortedIndices]
            x = x[:, sortedIndices]
            best_eval = evals[0]
            feasible = np.ones(lam)

            # 進化パス
            p_s = (1 - c_s) * p_s + np.sqrt(c_s * (2 - c_s) * mu_eff) * (z @ w_rank)
            normp_s = np.linalg.norm(p_s)
            # 学習率の計算
            eta_B = np.tanh((np.min([0.02 * np.sum(feasible), 3 * np.log(d)]) + 5) / (0.23 * d + 25))
            c_1 = c_1cma * (d - 5) / 6 * (np.sum(feasible) / lam)
            if normp_s >= eps:
                alpha = h_inv * min([1, np.sqrt(lam / d)]) * np.sqrt(sum(feasible) / lam)
                w_dist_hat = np.exp(alpha * np.sqrt(np.sum(z ** 2, axis=0))).reshape(lam, 1)
                a = 1 / (w_rank_hat.T @ w_dist_hat)[0][0]
                w_dist = w_rank_hat * w_dist_hat / (w_rank_hat.T @ w_dist_hat)[0][0] - (1 / lam)
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
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * wxm / sig  # 進化パス
            m = m + eta_m * wxm  # 中心ベクトルの更新
            # s, tの更新
            exY = np.append(y, p_c / D, axis=1)  # d x lam + 1
            yy = exY * exY  # d x lam + 1
            inyvbar = vbar.T @ exY  # 1 x lam + 1
            yvbar = exY * vbar  # d x lam+1
            gammav = 1 + normv2  # scalar
            normv4 = normv2 ** 2  # scalar
            vbarbar = vbar * vbar  # d x 1
            alphavd = np.min(
                [1, np.sqrt(normv4 + (2 * gammav - np.sqrt(gammav)) / np.max(vbarbar)) / (2 + normv2)])  # scalar
            t = exY * inyvbar - vbar * (inyvbar ** 2 + gammav) / 2  # d x lam+1
            b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2  # scalar
            H = np.ones((d, 1)) * 2 - (b + 2 * alphavd ** 2) * vbarbar  # d x 1
            invH = H ** (-1)  # d x 1
            s_step1 = yy - normv2 / gammav * (yvbar * inyvbar) + np.ones((d, lam + 1))  # d x lam + 1
            invbart = vbar.T @ t  # 1 x lam + 1
            s_step2 = s_step1 - alphavd / gammav * ((2 + normv2) * (t * vbar) - normv2 * vbarbar @ invbart)  # d x lam+1
            invHvbarbar = invH * vbarbar  # d x 1
            ins_step2invHvbarbar = invHvbarbar.T @ s_step2  # 1 x lam + 1
            s = (s_step2 * invH) - b / (
            1 + b * vbarbar.T @ invHvbarbar) * invHvbarbar @ ins_step2invHvbarbar  # d x lam + 1
            insvbarbar = vbarbar.T @ s  # 1 x lam + 1
            t = t - alphavd * ((2 + normv2) * (s * vbar) - vbar @ insvbarbar)  # d x lam+1
            # v, Dの更新
            exw = np.append(eta_B * w, np.array([l_c * c_1]).reshape(1, 1), axis=0)  # lam + 1 x 1
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
            g += 1

            print('g:{} best_eval:{}'.format(g, best_eval))
            data.append([g, best_eval])
            alg_stat = stop_condition(best_eval, g)
            if alg_stat == 1:
                is_success = True
                break
            elif alg_stat == -1:
                is_success = False
                break

        return is_success, data

    def _estimate_expansion_ratio(self):
        numitr = 1000
        expansion = 5
        for itr in range(numitr):
            square_expansion = expansion ** 2
            linearfitting = 0.24 * (self.d + 10)
            bunbo = (1 + square_expansion) * np.exp(square_expansion / 2) - linearfitting
            bunsi = (square_expansion + 3) * expansion * np.exp(square_expansion / 2)
            expansion = expansion - 0.5 * (bunbo / bunsi)
        return expansion
