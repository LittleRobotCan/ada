from bokeh.charts import BoxPlot, Histogram, output_file, save
from os.path import join
from scipy.stats import ranksums
from scipy import stats
import numpy as np
# correlations to target
# pca - does it increase correlations

def boxplot(df, target, value):
    name = y+"against"+value+"boxplot"
    p = BoxPlot(df, values=value, label=target,
                title=name)
    output_file(join('plots', name + ".html"))
    save(p)

def wilcox(df, target, value):
    a = df[df[target]==0][value]
    b = df[df[target]==1][value]
    results = ranksums(a, b)
    print y, results.pvalue

def distribution_plot(df, target, value):
    name = value + "distribution overlay"
    hist = Histogram(df, values=value, color=target, legend='top_right')
    output_file(join('plots', name+'.html'))
    save(hist)

def ttest(df, target, value, equal_var = True):
    a = df[df[target]==0][value]
    b = df[df[target]==1][value]
    results = stats.ttest_ind(a, b, equal_var=equal_var)
    return results[1]

def BH_correct(pvalues, alpha=0.05):
    pvalue_sorted = sorted(pvalues)
    n = len(pvalue_sorted)
    pval_max = np.max(pvalue_sorted)
    for i in range(len(pvalue_sorted)):
        pval = pvalue_sorted[i]
        j = i + 1
        HB = float(alpha) / (float(n) - float(j) + 1)
        if pval > HB:
            pval_max = pval
            break
        else:
            pass
    BH_stats = [True if pval < pval_max else False for pval in pvalues]
    return BH_stats


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    df = pd.read_csv('data/numerai_training_data.csv')

    # plot everything in boxplot, separating x, and y
    for i in range(1, 22):
        y = 'feature'+str(i)
        boxplot(df, 'target', y)
        print i
    # signal is very weak

    # plot overlayed distributions
    for i in range(1, 22):
        y = 'feature'+str(i)
        distribution_plot(df, 'target', y)
        print i
    # distribution plots also show no difference
    # data is normally distributed, use ttest

    # t test for normally distributed data
    np.random.seed(12345678)
    pvalues = []
    for i in range(1, 22):
        y = 'feature'+str(i)
        pvalues.append(ttest(df, 'target', y))

    pvalue_corrected = BH_correct(pvalues)
    for i in range(1, 22):
        print 'feature'+str(i), pvalue_corrected[i-1]
    # some significantly different features.
    """
    feature1 False
    feature2 True
    feature3 False
    feature4 True
    feature5 True
    feature6 False
    feature7 False
    feature8 False
    feature9 True
    feature10 True
    feature11 True
    feature12 True
    feature13 False
    feature14 False
    feature15 True
    feature16 False
    feature17 False
    feature18 True
    feature19 False
    feature20 True
    feature21 True
    """
