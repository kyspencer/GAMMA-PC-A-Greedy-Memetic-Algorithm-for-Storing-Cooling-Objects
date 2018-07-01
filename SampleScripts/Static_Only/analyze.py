# analyze.py
#   This file performs analyses on the results of optimization algorithms.
#   Author: Kristina Yancey Spencer
#   Date: June 10, 2016

import os
from glob import glob
from math import sqrt, ceil, floor
import items as itemmaker
import numpy as np
import pandas
import seaborn
from matplotlib import pyplot
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
from openpyxl import load_workbook

# Set environment for graphs
colors = ['#49ADA2', '#7797F4', '#C973F4', '#EF6E8B', '#FFCCCC', '#FFAA6C']
markers = ["o", "^", "D", "s", "v"]


def main():
    nop = 5
    runs = 20
    folders = ['/Users/gelliebeenz/Documents/Python/ObjectiveMethod/Static/GAMMA-PC/',
               '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/Static/NSGA-II/',
               '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/Static/MOMA/',
               '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/Static/MOMAD/',
               '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/Static/MOEPSO/',
               '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/Analysis/']
    files = ['ApproximationSet.csv', 'ApproximationSet.csv',
             'ApproximationSet.csv', 'ApproximationSet.csv',
             'ApproximationSet.csv']
    levels = ['SBSBPP500/']
    levellabels = ['500 Items']
    methods = ['GAMMA-PC', 'NSGA-II', 'MOMA', 'MOMAD', 'MOEPSO']
    solsindex = ['level', 'number', 'method']
    dfsolseries = []
    seaborn.set(font_scale=1.25)
    for lvl in range(len(levels)):
        if not glob(folders[nop] + levels[lvl]):
            os.mkdir(folders[nop] + levels[lvl])
        ietas = []
        solsdiversity = []
        cover_matrices = []
        spreadindicators = []
        for r in range(runs):
            print('Analyzing Experiment{0:02d}'.format(r + 1))
            runfolds = listfolders(r, lvl, nop, folders, levels)
            paretoplot(nop, runfolds, files, methods)
            # getbininfo(nop, runfolds, methods)
            numdiffsols, ieta, cover, spread = evalperformance(nop, runfolds, files, methods)
            for opal in range(nop):
                solsdiversity.append([levellabels[lvl], numdiffsols[opal], methods[opal]])
                spreadindicators.append([r+1, spread[opal], methods[opal]])
            ietas.append(ieta)
            cover_matrices.append(cover)
        df_new = combineindicators(nop, solsdiversity, solsindex, ietas,
                                   cover_matrices, spreadindicators,
                                   folders[nop] + levels[lvl], methods)
        dfsolseries.append(df_new)
    gettotalindicators(dfsolseries, solsindex, folders[nop])


def paretoplot(nop, folders, files, methods):
    # This function generates a plot of the Pareto Fronts.
    # input: number of algorithms to compare, file locations
    # output: 3D Pareto Front and 2D Scatter plots
    print('Making Pareto Front plots.')
    data, opcat = getparetofront(nop, folders, files)
    for opal in range(nop):
        method = []
        for m in range(len(data[opal])):
            method.append(methods[opal])
        data[opal]['Method'] = method
    # Plot 0: 3D Pareto Front Plot
    # -------------------------------------------------------------------------
    seaborn.set_style('whitegrid')
    plotname0 = folders[nop] + 'ParetoPlot3D.eps'
    plot0 = pyplot.figure().gca(projection='3d')
    for opal in range(nop):
        plot0.scatter(data[opal]['No. of Bins'], data[opal]['Max. Bin Height'],
                      data[opal]['Avg. Bin Weight'], c=colors[opal], marker=markers[opal],
                      label=methods[opal])
    plot0.set_xlabel(opcat[0], labelpad=10)
    plot0.set_ylabel(opcat[1], labelpad=10)
    start, end = plot0.get_ylim()
    plot0.yaxis.set_ticks(np.arange(start, end, 100))
    plot0.set_zlabel(opcat[2], labelpad=10)
    plot0.legend(bbox_to_anchor=(0.1, 0, 1, 1), ncol=nop)
    plot0.view_init(20, 45)
    pyplot.savefig(plotname0, format='eps', dpi=2000)
    pyplot.close()
    # Plot 1: 2D Pareto Front Plots
    # -------------------------------------------------------------------------
    seaborn.set_style('darkgrid')
    objectives = [[0, 1], [0, 2], [1, 2]]
    for i, j in objectives:
        plotname1 = folders[nop] + 'ParetoFront' + str(i+1) + str(j+1) + '.eps'
        plot1 = pyplot.figure(dpi=2000)
        ax1 = plot1.add_subplot(111)
        for opal in range(nop):
            x = getcolumn(i, opal, data)
            y = getcolumn(j, opal, data)
            ax1.scatter(x, y, s=40, c=colors[opal], marker='o', label=methods[opal])
        ax1.set_xlabel(opcat[i])
        ax1.set_ylabel(opcat[j])
        ax1.legend(loc='upper right', frameon=True)
        pyplot.savefig(plotname1, format='eps')
        pyplot.close()
    # Plot 2: Scatter Matrix Plot
    # ------------------------------------------------------------------------
    plotname2 = folders[nop] + 'ParetoPlot.eps'
    dataset = pandas.concat(data, keys=methods)
    scat = seaborn.PairGrid(dataset, hue='Method', palette=seaborn.color_palette(colors),
                            hue_kws={"marker": markers})
    scat = scat.map_diag(pyplot.hist)
    scat = scat.map_offdiag(pyplot.scatter, linewidths=0.5, edgecolor="w", s=40)
    for ax in scat.axes.flat:
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda xax, p: format(int(xax))))
        pyplot.setp(ax.get_xticklabels(), rotation=45)
    scat.add_legend(title=None, frameon=True)
    scat.fig.get_children()[-1].set_bbox_to_anchor((0.995, 0.925, 0, 0))
    pyplot.savefig(plotname2, format='eps', dpi=4000)
    pyplot.close()


def getcolumn(index, opal, data):
    # This function figures out which characteristic is being selected.
    if index == 0:
        column = data[opal]['No. of Bins']
    elif index == 1:
        column = data[opal]['Max. Bin Height']
    else:
        column = data[opal]['Avg. Bin Weight']
    return column


def getbininfo(nop, folders, methods):
    writer = pandas.ExcelWriter(folders[nop] + 'IndividualBinInfo.xlsx', engine='openpyxl')
    for opal in range(nop):
        itemfile = glob(folders[opal] + '*.txt')
        binc, binh, items = itemmaker.makeitems(itemfile[0])
        info = itemmaker.getiteminfo(items)
        xs, ys = getxys(folders[opal], len(items))
        for f in range(len(xs)):
            xmatrix = np.zeros((len(items), len(items)))
            for i in range(len(items)):
                for j in range(len(items)):
                    xmatrix[i, j] = xs[f].get_value(i, j, takeable=True)
            binws = np.dot(xmatrix, info[0])
            binhs = np.dot(xmatrix, info[1])
            binws_series = []
            binhs_series = []
            end = np.where(ys[f] == 0)
            for i in range(end[0][0]):
                binws_series.append(binws[i, 0])
                binhs_series.append(binhs[i, 0])
            if f == 0:
                dfw = pandas.DataFrame(binws_series)
                dfh = pandas.DataFrame(binhs_series)
            else:
                df1 = pandas.DataFrame(binws_series)
                df2 = pandas.DataFrame(binhs_series)
                dfw = pandas.concat([dfw, df1], ignore_index=True, axis=1)
                dfh = pandas.concat([dfh, df2], ignore_index=True, axis=1)
        dfw.to_excel(writer, sheet_name=methods[opal]+'_weight')
        dfh.to_excel(writer, sheet_name=methods[opal]+'_height')
        writer.save()


def getbinplots(nop, folders, methods):
    # This function sorts through the data to make plots about
    # individual bin characteristics.
    print('Making Bin Weights plot.')
    wplot = pyplot.figure()
    ax = wplot.add_subplot(111)
    dfseries = []
    for opal in range(nop):
        itemfile = glob(folders[opal] + '*.txt')
        binc, binh, items = itemmaker.makeitems(itemfile[0])
        info = itemmaker.getiteminfo(items)
        xs, ys = getxys(folders[opal], len(items))
        xplot = []
        for f in range(len(xs)):
            xplot.append(np.zeros((len(items), len(items))))
            for i in range(len(items)):
                for j in range(len(items)):
                    xplot[f][i, j] = xs[f].get_value(i, j, takeable=True)
        sols_wavg, sols_wmax = binwsplot(xplot, ys, info[0], colors[opal], methods[opal])
        df1 = pandas.DataFrame({'Solution Number': range(1, 1 + len(sols_wavg)),
                                'Weight': sols_wavg, 'Statistic': 'Average',
                                'Method': methods[opal]})
        df2 = pandas.DataFrame({'Solution Number': range(1, 1 + len(sols_wavg)),
                                'Weight': sols_wmax, 'Statistic': 'Maximum',
                                'Method': methods[opal]})
        dfopal = pandas.concat((df1, df2), axis=0)
        dfseries.append(dfopal)
    fig = folders[nop] + 'BinWeight_comparison.eps'
    ax.set_xlim(xmin=0)
    ax.set_xlabel('Bin Weight')
    ax.legend(loc='upper right', frameon=True)
    pyplot.savefig(fig, format='eps', dpi=2000)
    pyplot.close()
    dfstats = pandas.concat(dfseries, axis=0)
    getbinstatsplot(dfstats, folders[nop])


def binwsplot(xpats, ys, weights, color, method):
    # This function plots the bin weight distributions for a given
    # x and y combo.
    join = []
    for f in range(len(xpats)):
        binweights = np.dot(xpats[f], weights)
        end = np.where(ys[f] == 0)
        join.append(binweights[:end[0][0]])
    sols_wavg = []
    sols_wmax = []
    for m in range(len(join)):
        sols_wavg.append(np.average(join[m]))
        sols_wmax.append(np.amax(join[m]))
    sols_wavg, sols_wmax = zip(*sorted(zip(sols_wavg, sols_wmax)))
    binws = np.concatenate(join, axis=0)
    meanbin = np.average(binws)
    medbin = np.median(binws)
    setlabel = method + ', mean={0:4.1f}, median={1:4.1f}'.format(meanbin, medbin)
    seaborn.distplot(binws, norm_hist=False, color=color, label=setlabel)
    return sols_wavg, sols_wmax


def getbinstatsplot(dfstats, folder):
    pyplot.figure()
    kws = dict(s=70, linewidth=1, edgecolor="w")
    plotstats = seaborn.lmplot('Solution Number', 'Weight', data=dfstats, hue='Statistic',
                               markers=["o", "^"], col='Method', fit_reg=False, legend_out=True,
                               scatter_kws=kws)
    plotstats.set(xlim=(0, None))
    plotstats.add_legend(frameon=True)
    figstats = folder + 'BinWeights_bySolution.eps'
    pyplot.savefig(figstats, format='eps', dpi=2000)
    pyplot.close()


def evalperformance(nop, folders, files, methods):
    # This function automates the task of calculating performance indicators
    # for a specific experiment and level.
    data, opcat = getparetofront(nop, folders, files)
    icols = []
    numdiffsols = np.zeros(nop, dtype=int)
    for opal in range(nop):
        icols.append((methods[opal], 'f4'))
        numdiffsols[opal] = len(data[opal])
    # Make binary indicator
    i_etamatrix = np.zeros((nop,), dtype=icols)
    for op1 in range(nop):
        for op2 in range(nop):
            i_etamatrix[op2][op1] = getbinindicators(data[op1], data[op2])
    i_etadf = pandas.DataFrame(i_etamatrix, index=methods)
    # Make coverage indicator
    coveragematrix = calc_coverage(nop, data)
    df_cover = pandas.DataFrame(coveragematrix, index=methods)
    # Make diversity indicator
    spread = getspread(nop, data, opcat)
    # Write data to Indicators.xlsx
    writer = pandas.ExcelWriter(folders[nop] + 'Indicators.xlsx')
    i_etadf.to_excel(writer, sheet_name='BinaryIndicators')
    df_cover.to_excel(writer, sheet_name='Coverage')
    writer.save()
    return numdiffsols, i_etamatrix, coveragematrix, spread


def getbinindicators(opal_a, opal_b):
    # This function calculates the binary eta-indicators I_eta(A,B)
    # and I_eta(B,A) where A is the first method and B is the second method.
    # Reference: Zitzler, 2003
    eta12 = np.zeros((len(opal_a), len(opal_b)))
    for a in range(len(opal_a)):
        for b in range(len(opal_b)):
            ratios = [(opal_a.iloc[a][0]) / (opal_b.iloc[b][0]),
                      (opal_a.iloc[a][1]) / (opal_b.iloc[b][1]),
                      (opal_a.iloc[a][2]) / (opal_b.iloc[b][2])]
            eta12[a, b] = np.amax(ratios)
    eta2 = np.zeros(len(opal_b))
    for b in range(len(opal_b)):
        eta2[b] = np.amin(eta12[:, b])
    i_ab = np.amax(eta2)
    return i_ab


def calc_coverage(nop, data):
    # Make coverage indicator
    coveragematrix = np.zeros((nop, nop))
    for a in range(nop):
        for b in range(nop):
            if a == b:
                coveragematrix[b, a] = 1
            else:
                n_bfront = len(data[b])  # fraction denominator
                n_bcovered = 0           # fraction numerator
                for mb in range(n_bfront):
                    v = np.matrix([data[b].get_value(mb, 0, takeable=True),
                                   data[b].get_value(mb, 1, takeable=True),
                                   data[b].get_value(mb, 2, takeable=True)])
                    for ma in range(len(data[a])):
                        u = np.matrix([data[a].get_value(ma, 0, takeable=True),
                                       data[a].get_value(ma, 1, takeable=True),
                                       data[a].get_value(ma, 2, takeable=True)])
                        if dom(u, v):
                            n_bcovered += 1
                            break
                coveragematrix[b, a] = float(n_bcovered) / n_bfront
    return coveragematrix


def getspread(nop, data, opcat):
    # This module finds the spread indicator values for each algorithm
    # and then normalizes them to the largest value.
    spreadvals = np.zeros(nop)
    for opal in range(nop):
        spreadvals[opal] = calcspread(data[opal], opcat)
    normfactor = np.max(spreadvals)
    spread = spreadvals / normfactor
    return spread


def calcspread(opaldata, opcat):
    # This module calculates the spread indicator from an approximate set
    dsum = 0
    for o in range(len(opcat)):
        maxval = opaldata.nlargest(1, opcat[0]).get_value(0, 0, takeable=True)
        minval = opaldata.nsmallest(1, opcat[0]).get_value(0, 0, takeable=True)
        dsum += (maxval - minval)**2
    dspread = sqrt(dsum)
    return dspread


def getdeltas(nop, data, opcat):
    # This function calculates the delta values for the Deb diversity indicator
    deltas = np.zeros(nop)
    for opal in range(nop):
        front0 = data[opal].sort_values(opcat[2])
        dis = np.zeros(len(front0) - 1)
        for m in range(len(front0) - 1):
            a = np.zeros(3)
            b = np.zeros(3)
            for f in range(3):
                a[f] = front0.get_value(m, f, takeable=True)
                b[f] = front0.get_value(m + 1, f, takeable=True)
            dis[m] = np.linalg.norm(a - b)
        davg = np.average(dis)
        sumd = 0
        for m in range(len(front0) - 1):
            sumd += abs(dis[m] - davg)
        deltas[opal] = sumd / len(front0)
    deltadiffs = np.identity(nop)
    for op1 in range(nop):
        for op2 in range(nop):
            if op1 != op2:
                deltadiffs[op1, op2] = deltas[op1] / deltas[op2]
    return deltas, deltadiffs


def combineindicators(nop, numsols, solsindex, ietas, covers, spreads, folder, methods):
    # This function combines the results given by getbinindicators(A, B)
    # into a simple histogram.
    print('Combining indicators.')
    df_numsols = pandas.DataFrame(numsols)
    df_numsols.columns = solsindex
    df_spread = pandas.DataFrame(spreads)
    df_spread.columns = ['Run', 'Spread', 'Method']
    makeperformxlsx(df_numsols, df_spread, folder)
    figlabel = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)',
                '(h)', '(i)', '(j)']
    part = 0
    for a in range(nop):
        for b in range(a+1, nop):
            makeietaplots(a, b, ietas, folder, figlabel[part], methods)
            etabetter = getetabetter(a, b, ietas, methods)
            plotetabetter(nop, etabetter, a, b, figlabel[part], folder, methods)
            coverbetter = getcoverbetter(a, b, covers, methods)
            plotcoverbetter(nop, coverbetter, a, b, figlabel[part], folder, methods)
            part += 1
    makespreadplot(df_spread, folder)
    return df_numsols


def makeperformxlsx(numsols_df, df_spread, folder):
    # Write data to PerformanceMetrics.xlsx
    writer = pandas.ExcelWriter(folder + 'PerformanceMetrics.xlsx')
    numsols_df.to_excel(writer, sheet_name='NumberSolutions')
    df_spread.to_excel(writer, sheet_name='Spread Indicators')
    writer.save()


def gettotalindicators(dfsolseries, solsindex, folder):
    # This function plots the number of solutions found for all levels.
    df_sols = pandas.concat(dfsolseries, axis=0)
    plt = seaborn.barplot(x=solsindex[0], y=solsindex[1], hue=solsindex[2], data=df_sols,
                          palette=seaborn.color_palette(colors), capsize=.2)
    plt.set(xlabel='Experiment Level', ylabel='Avg. # of Approximate Solutions Found')
    pyplot.savefig(folder + 'RangeofSolutions.eps', format='eps', dpi=2000)


def makeietaplots(a, b, ietas, folder, xlabel, methods):
    # This function plots the distribution of I_etas of A into B
    plotname = folder + 'Epsilon_' + methods[a] + 'vs' + methods[b] + '.eps'
    df_epsilon = make_df_epsilon(a, b, ietas, methods)
    seaborn.set_context('talk', font_scale=1.5)
    pyplot.figure(figsize=(8, 6))
    plot = seaborn.boxplot(x='Comparison', y='I-Epsilon', data=df_epsilon,
                           palette=[colors[a], colors[b]])
    plot.axes.set_xlabel(xlabel)
    plot.axes.set_ylabel('$I_\epsilon(A,B)$', labelpad=10)
    plot.axes.set(ylim=(get_ylimits(ietas)))
    pyplot.savefig(plotname, format='eps', dpi=2000)
    pyplot.close()


def make_df_epsilon(a, b, ietas, methods):
    # This function transforms ietas into a dataframe for the boxplot
    list_epsilon = []
    for i in range(len(ietas)):
        list_epsilon.append({
            'Comparison': '({0},{1})'.format(methods[a], methods[b]),
            'I-Epsilon': ietas[i][b][a]
        })
        list_epsilon.append({
            'Comparison': '({0},{1})'.format(methods[b], methods[a]),
            'I-Epsilon': ietas[i][a][b]
        })
    df_epsilon = pandas.DataFrame(list_epsilon)
    return df_epsilon


def get_ylimits(ietas):
    # This function finds the minimum and maximum values in ietas and
    # returns y-limits for box plots
    ieta_values = []
    for i in range(len(ietas)):
        ieta_values.extend(list(ietas[i]))
    min_values = [min(lst) for lst in ieta_values]
    max_values = [max(lst) for lst in ieta_values]
    ymin = floor(min(min_values)) - 0.1
    ymax = ceil(max(max_values))
    return ymin, ymax


def plotetabetter(nop, better, a, b, xlabel, folder, methods):
    plotname = folder + 'Ieta_' + methods[a] + 'vs' + methods[b] + '.eps'
    plotcolors = [colors[a], colors[b], colors[nop]]
    metha = 0
    methb = 0
    nope = 0
    for eta in better:
        if eta is methods[a]:
            metha += 1
        elif eta is methods[b]:
            methb += 1
        else:
            nope += 1
    betterdict = {'Method': [methods[a], methods[b], 'Neither'],
                  'Number of Experiments': [metha, methb, nope]}
    bdf = pandas.DataFrame(betterdict)
    pyplot.figure()
    seaborn.set_context('talk', font_scale=1.5)
    betplot = seaborn.barplot('Method', 'Number of Experiments', data=bdf,
                              ci=None, palette=plotcolors)
    betplot.set_xlabel(xlabel)
    betplot.set_ylabel('No. Better by $I_\epsilon(A,B)$', labelpad=10)
    betplot.set(ylim=(0, len(better)))
    pyplot.savefig(plotname, format='eps', dpi=2000)
    pyplot.close()


def getetabetter(a, b, ietas, methods):
    better = []
    for i in range(len(ietas)):
        if etabetter(a, b, ietas[i]):
            better.append(methods[a])
        elif etabetter(b, a, ietas[i]):
            better.append(methods[b])
        else:
            better.append('Neither')
    return better


def etabetter(a, b, ieta):
    # Interpretation function from Zitzler (2003) for Binary Indicator
    aintob = ieta[b][a]
    bintoa = ieta[a][b]
    return aintob <= 1 and bintoa > 1


def plot_coverage_comparison(a, b, covers, folder, xlabel, methods):
    # This function plots box plots showing the coverage indicator comparison
    plotname = folder + 'Cover_' + methods[a] + 'vs' + methods[b] + '.eps'
    df_cover = make_df_cover(a, b, covers, methods)
    seaborn.set_context('talk', font_scale=1.5)
    pyplot.figure(figsize=(8, 6))
    plot = seaborn.boxplot(x='Comparison', y='I-Cover', data=df_cover, palette=[colors[a], colors[b]])
    plot.axes.set_xlabel(xlabel)
    plot.axes.set_ylabel('$I_C(A,B)$', labelpad=10)
    plot.axes.set(ylim=(0, 1))
    pyplot.savefig(plotname, format='eps', dpi=2000)
    pyplot.close()


def make_df_cover(a, b, cover_matrices, methods):
    # This function transforms cover_matrices into a dataframe for the boxplot
    list_cover = []
    for i in range(len(cover_matrices)):
        list_cover.append({
            'Comparison': '({0},{1})'.format(methods[a], methods[b]),
            'I-Cover': cover_matrices[i][b][a]
        })
        list_cover.append({
            'Comparison': '({0},{1})'.format(methods[b], methods[a]),
            'I-Cover': cover_matrices[i][a][b]
        })
    df_cover = pandas.DataFrame(list_cover)
    return df_cover


def plotcoverbetter(nop, better, a, b, xlabel, folder, methods):
    plotname = folder + 'Coverage_' + methods[a] + 'vs' + methods[b] + '.eps'
    plotcolors = [colors[a], colors[b], colors[nop]]
    metha = 0
    methb = 0
    nope = 0
    for cover in better:
        if cover is methods[a]:
            metha += 1
        elif cover is methods[b]:
            methb += 1
        else:
            nope += 1
    betterdict = {'Method': [methods[a], methods[b], 'Neither'],
                  'Number of Experiments': [metha, methb, nope]}
    bdf = pandas.DataFrame(betterdict)
    pyplot.figure()
    seaborn.set_context('talk', font_scale=1.5)
    betplot = seaborn.barplot('Method', 'Number of Experiments', data=bdf, ci=None,
                              palette=plotcolors)
    betplot.set_xlabel(xlabel)
    betplot.set_ylabel('No. Better by $I_C(A,B)$', labelpad=10)
    betplot.set(ylim=(0, len(better)))
    pyplot.savefig(plotname, format='eps', dpi=2000)
    pyplot.close()


def getcoverbetter(a, b, covers, methods):
    better = []
    for i in range(len(covers)):
        if coverbetter(a, b, covers[i]):
            better.append(methods[a])
        elif coverbetter(b, a, covers[i]):
            better.append(methods[b])
        else:
            better.append('Neither')
    return better


def coverbetter(a, b, covermatrix):
    # Interpretation function from Zitzler (2003) for coverage indicator
    aintob = covermatrix[b, a]
    bintoa = covermatrix[a, b]
    return aintob == 1 and bintoa < 1


def makespreadplot(df_spread, folder):
    # This module plots the spread indicators
    # x-axis: experiment number
    # y-axis: Spread value
    kws = dict(s=70, linewidth=1, edgecolor="w")
    plotname = folder + 'SpreadIndicators'
    seaborn.set_context('paper', font_scale=1.25)
    plot = seaborn.lmplot(x='Run', y='Spread', data=df_spread, hue='Method',
                          palette=seaborn.color_palette(colors), size=5,
                          aspect=1.5, legend_out=False, fit_reg=False,
                          scatter_kws=kws, markers=markers)
    plot.set(xlabel='Experiment Number', ylabel='Spread Indicator (Normalized)')
    plot.set(xlim=(0, 21))
    plot.set(yscale='log')
    plot.add_legend(frameon=True)
    pyplot.savefig(plotname + '.eps', format='eps')
    pyplot.savefig(plotname + '.pdf', format='pdf')
    pyplot.close()


def makedeltaplots(nop, deltas, folder, methods):
    # This function plots the differences in deltas for each experimental level
    for a in range(nop):
        for b in range(nop):
            if a != b:
                plotname = folder + 'DeltaDifferentials_' + methods[a] + '_into_' + methods[b] + '.eps'
                distab = np.zeros(len(deltas))
                for i in range(len(deltas)):
                    distab[i] = deltas[i][a, b]
                seaborn.set_context('talk', font_scale=1.5)
                pyplot.figure(figsize=(8, 6))
                plot = seaborn.distplot(distab, kde=False, color=colors[a])
                plot.axes.set_title('({0},{1})'.format(methods[a], methods[b]), fontsize=28)
                start, end = plot.get_xlim()
                if abs(end-1) > abs(1-start):
                    newend = round((end + 0.05), 1)
                    plot.set_xlim(2 - newend, newend)
                else:
                    newstart = round((start - 0.05), 1)
                    plot.set_xlim(newstart, 2 - newstart)
                pyplot.savefig(plotname, format='eps', dpi=2000)
                pyplot.close()


def getparetofront(nop, folders, files):
    # This function creates the Pareto Front dataframe.
    data = []
    opcat = ['No. of Bins', 'Max. Bin Height', 'Avg. Bin Weight']
    for opal in range(nop):
        data.append(pandas.read_csv(folders[opal] + files[opal], index_col=0))
        data[opal].columns = opcat
    return data, opcat


def getxys(folder, nitems):
    # This function sorts through the x and y text files and returns x and y
    # in a dataframe.
    flag = 'variables/'
    xfiles = glob(folder + flag + '*_x.txt')
    yfiles = glob(folder + flag + '*_y.txt')
    solids = []
    for f in range(len(xfiles)):
        idstr = xfiles[f][xfiles[f].find(flag) + len(flag):xfiles[f].find('_')]
        solids.append(int(idstr))
    xfiles = [x for (solid, x) in sorted(zip(solids, xfiles))]
    yfiles = [y for (solid, y) in sorted(zip(solids, yfiles))]
    xs = []
    ys = []
    for f in range(len(xfiles)):
        x = pandas.read_csv(xfiles[f], sep=' ', header=None, names=range(nitems), skiprows=1)
        y = pandas.read_csv(yfiles[f], sep=' ', header=None, skiprows=1)
        xs.append(x)
        ys.append(y)
    return xs, ys


def listfolders(r, lvl, nop, folders, levels):
    # This function combines the folder names together.
    runfolds = []
    for opal in range(nop):
        drcty = folders[opal] + levels[lvl]
        runfolds.append(drcty + 'Experiment{0:02d}/'.format(r+1))
    runfolds.append(folders[nop] + levels[lvl] + 'Experiment{0:02d}/'.format(r+1))
    pathname = os.path.dirname(runfolds[nop])
    if glob(pathname) == []:
        os.mkdir(pathname)#
    return runfolds


def dom(u, v):
    # Determines if fitness vector u dominates fitness vector v
    # This function assumes a minimization problem.
    # For u to dominate v, every fitness value must be either
    # equal to or less than the value in v AND one fitness value
    # must be less than the one in v
    equaltest = np.allclose(u, v)
    if equaltest is True:
        # If u == v then nondominated
        return False
    # less_equal returns boolean for each element u[i] <= v[i]
    domtest = np.less_equal(u, v)
    return np.all(domtest)


if __name__ == '__main__':
    main()
