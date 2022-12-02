import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterExponent


def graph_log_loss(log_loss):
    """Log loss visualization

    Inputs: 
        log_loss (Numpy Array): array of log loss to visualize

    Returns: 
        ax (Matplotlib Graph)
    """
    pd.set_option("display.precision", 8)

    df = pd.DataFrame(log_loss.T, columns = ['ridge','alpha_0.1','alpha_0.2',
                                            'alpha_0.3','alpha_0.4','alpha_0.5',
                                            'alpha_0.6','alpha_0.7','alpha_0.8',
                                            'alpha_0.9', 'lasso'])

    df.insert(0,'lambda', [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001])

    df_melt = pd.melt(df, id_vars=['lambda'], 
                          value_vars=['ridge', 'alpha_0.1',
                                      'alpha_0.2', 'alpha_0.3', 'alpha_0.4', 
                                      'alpha_0.5', 'alpha_0.6', 'alpha_0.7',
                                      'alpha_0.8', 'alpha_0.9', 'lasso'],
                          var_name='model_type',
                          value_name='avg_err' )

    sns.set(rc={'figure.figsize':(11.7,8.27)})

    ax = sns.lineplot(x='lambda', y='avg_err', data=df_melt, hue='model_type')
    ax.set_xscale('log')
    ax.set_title('Average Error by Model Specification', fontsize=18)
    ax.set_ylabel('Average Error Rate')

    return ax
