from os import path
from plot_image import get_path


def write_experiment_result(initial_mu, P, mu, cov):
    with open(path.join(get_path('作业文档'), 'result.tex'), 'w') as f:
        result = r"""\[
    \hat{\mu}_1^{(0)} =  \left[
        \begin{matrix}
            ${mu_1_1} \\
            ${mu_1_2}
        \end{matrix}
        \right]\text{ , }
    \hat{\mu}_2^{(0)} =  \left[
        \begin{matrix}
            ${mu_2_1} \\
            ${mu_2_2}
        \end{matrix}
        \right]\text{ , }
    \hat{\mu}_3^{(0)} = \left[
        \begin{matrix}
            ${mu_3_1} \\
            ${mu_3_2}
        \end{matrix}
        \right]
\]
\[
    \hat{\Sigma}_1^{(0)} = \hat{\Sigma}_2^{(0)} = \hat{\Sigma}_3^{(0)} = \left[
        \begin{matrix}
            1 & 0 \\
            0 & 1
        \end{matrix}
        \right]
\]

执行1000次迭代后，得到的结果为:

\[
    \hat{P}_1^{(1000)} = ${p_h_1} \text{ , } \hat{P}_2^{(1000)} = ${p_h_2} \text{ , } \hat{P}_3^{(1000)} = ${p_h_3}
\]
\[
    \hat{\mu}_1^{(1000)} =  \left[
        \begin{matrix}
            ${mu_h_1_1} \\
            ${mu_h_1_2}
        \end{matrix}
        \right]\text{ , }
    \hat{\mu}_2^{(1000)} =  \left[
        \begin{matrix}
            ${mu_h_2_1} \\
            ${mu_h_2_2}
        \end{matrix}
        \right]\text{ , }
    \hat{\mu}_3^{(1000)} = \left[
        \begin{matrix}
            ${mu_h_3_1} \\
            ${mu_h_3_2}
        \end{matrix}
        \right]
\]
\[
    \hat{\Sigma}_1^{(1000)} = \left[
        \begin{matrix}
            ${si_h_1_1_1} & ${si_h_1_1_1} \\
            ${si_h_1_2_1} & ${si_h_1_2_2}
        \end{matrix}
        \right] \text{ , }
    \hat{\Sigma}_2^{(1000)} = \left[
        \begin{matrix}
            ${si_h_2_1_1} & ${si_h_2_1_1} \\
            ${si_h_2_2_1} & ${si_h_2_2_2}
        \end{matrix}
        \right] \text{ , }
    \hat{\Sigma}_3^{(1000)} = \left[
        \begin{matrix}
            ${si_h_3_1_1} & ${si_h_3_1_1} \\
            ${si_h_3_2_1} & ${si_h_3_2_2}
        \end{matrix}
        \right]
\]
        """
        dic = {}
        for i in range(3):
            for j in range(2):
                dic[f'mu_{i + 1}_{j + 1}'] = initial_mu[i][j]
                dic[f'mu_h_{i + 1}_{j + 1}'] = mu[i][j]
            dic[f'p_h_{i + 1}'] = P[i]
            for j in range(2):
                for k in range(2):
                    dic[f'si_h_{i + 1}_{j + 1}_{k + 1}'] = cov[i][j][k]

        for key, value in dic.items():
            result = result.replace('${' + key + '}', f'{value:.7f}')

        f.write(result)
