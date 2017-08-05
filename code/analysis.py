from bokeh.charts import BoxPlot, output_file, save
from os.path import join
from scipy.stats import ranksums

# correlations to target
# pca - does it increase correlations

def boxplot(df, x, y):
    name = y+"against"+x+"boxplot"
    p = BoxPlot(df, values=y, label=x,
                title=name)
    output_file(join('plots', name + ".html"))
    save(p)

def wilcox(df, x, y):
    a = df[df[x]==0][y]
    b = df[df[x]==1][y]
    results = ranksums(a, b)
    print y, results.pvalue

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/numerai_training_data.csv')

    # plot everything in boxplot, separating x, and y
    for i in range(1, 22):
        y = 'feature'+str(i)
        boxplot(df, 'target', y)
        print i
    # signal is very weak

    # try wilcoxon
    for i in range(1, 22):
        y = 'feature'+str(i)
        wilcox(df, 'target', y)
    """
    feature1 0.717926063426
    feature2 1.35954204318e-05
    feature3 0.000703331291827
    feature4 4.69779351787e-07
    feature5 5.78035434422e-06
    feature6 0.142169105993
    feature7 0.00849296737596
    feature8 0.715000838903
    feature9 3.84283403425e-05
    feature10 3.48624241084e-08
    feature11 1.41219799387e-06
    feature12 1.3782671143e-10
    feature13 0.00942932940825
    feature14 0.156491406836
    feature15 4.34607662964e-05
    feature16 0.707818436058
    feature17 0.0108523729628
    feature18 3.46405199169e-05
    feature19 0.00278570085803
    feature20 1.01728635502e-09
    feature21 1.58744789763e-10
    
    """
    # some features turn out to be significantly different
    # the sets are not drawn from the same distribution

    # 