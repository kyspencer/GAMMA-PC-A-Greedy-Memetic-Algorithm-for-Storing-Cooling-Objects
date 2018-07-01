# analyze.py
#   This file performs analyses on the results of optimization algorithms.
#   Author: Kristina Yancey Spencer
#   Date: June 10, 2016

from __future__ import print_function
import h5py
import numpy as np
import os
import pandas
import seaborn
from coolcookies import makeobjects
from glob import glob
from math import ceil, sqrt
from matplotlib import pyplot
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
from openpyxl import load_workbook

# Set environment for graphs
colors = ['#49ADA2', '#7797F4', '#C973F4', '#EF6E8B', '#FFAA6C']
markers = ["o", "^", "D", "s"]
figlabel = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


def main():
    nop = 3
    runs = 20
    basefolder = '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/TimeDependent/'
    folders = [basefolder + 'GAMMA-PC/', basefolder + 'NSGA-II/',
               basefolder + 'MOMA/',
               '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/Analysis/']
    files = ['ApproximationSet.csv', 'ApproximationSet.csv', 'ApproximationSet.csv']
    n = [1000]
    batchsize = [100]
    levels = ['Cookies1000/']
    levellabels = ['1000 Cookies']
    methods = ['GAMMA-PC', 'NSGA-II', 'MOMA']
    solsindex = ['Level', 'Number of Solutions', 'Method']
    analyze_sets(nop, runs, folders, files, n, batchsize, levels, levellabels,
                 methods, solsindex)


def analyze_sets(nop, runs, folders, files, n, batchsize, levels, levellabels,
                 methods, solsindex, wpareto=False):
    dfsolseries = []
    opcat = ['No. of Bins', 'Avg. Initial Bin Heat (W)', 'Max. Time to Move (s)']
    seaborn.set(font_scale=1.25)
    for lvl in range(len(levels)):
        if not glob(folders[-1] + levels[lvl]):
            os.mkdir(folders[-1] + levels[lvl])
        datacollect = gather_solutions(runs, lvl, n, batchsize, nop, files,
                                       folders, levels, methods, opcat,
                                       wpareto=wpareto)
        df_numsols = combineindicators(n[lvl], nop, runs, solsindex, datacollect,
                                       opcat, folders[-1] + levels[lvl],
                                       methods, wpareto=wpareto)
        dfsolseries.append(df_numsols)
    gettotalindicators(dfsolseries, solsindex, folders[nop])


def gather_solutions(runs, lvl, n, batchsize, nop, files, folders, levels,
                     methods, opcat, wpareto=False):
    # Uncomment makeobjects if you need to check constraints
    cookies = makeobjects(n[lvl], batchsize[lvl], folders[0] + levels[lvl] +
                          'Experiment01/Cookies{0:d}.txt'.format(n[lvl]))
    print('Gathering Data')
    datacollect = []
    for r in range(runs):
        runfolds = listfolders(r, lvl, nop, folders, levels)
        # checkforviolations(n[lvl], runfolds, cookies, methods)
        data = getparetofront(nop, opcat, runfolds, files, methods, r)
        datacollect.extend(data)
        paretoplot(data, opcat, nop, runfolds[-1], methods, colors)
        getbininfo(cookies, nop, runfolds, methods)
    if wpareto:
        pareto = pandas.read_csv(folders[-2] + files[0], index_col=0)
        pareto.columns = opcat
        pareto['Method'] = 'Pareto'
        pareto['Experiment'] = 1
        datacollect.append(pareto)
    return datacollect


def getparetofront(nop, opcat, folders, files, methods, r):
    # This function creates the Pareto Front dataframe.
    data = []
    for opal in range(nop):
        data.append(pandas.read_csv(folders[opal] + files[opal], index_col=0))
        data[opal].columns = opcat
        data[opal]['Method'] = methods[opal]
        data[opal]['Experiment'] = r + 1
    return data


def paretoplot(data, opcat, nop, folder, methods, color_choices, ignore_method=False):
    # This function generates a plot of the Pareto Fronts.
    # input: number of algorithms to compare, file locations
    # output: 3D Pareto Front and 2D Scatter plots
    print('Making Pareto Front plots.')
    # Plot 0: 3D Pareto Front Plot
    # -------------------------------------------------------------------------
    seaborn.set_style('whitegrid')
    plotname0 = folder + 'ParetoPlot3D'
    plot0 = pyplot.figure().gca(projection='3d')
    for opal in range(nop):
        plot0.scatter(data[opal][opcat[0]], data[opal][opcat[1]],
                      data[opal][opcat[2]], c=color_choices[opal], label=methods[opal])
    plot0.set_xlabel(opcat[0], labelpad=10)
    plot0.set_ylabel(opcat[1], labelpad=10)
    start, end = plot0.get_ylim()
    plot0.yaxis.set_ticks(np.arange(start, end, 100))
    plot0.set_zlabel(opcat[2], labelpad=10)
    plot0.legend(bbox_to_anchor=(-0.25, -1, 1, 1), ncol=nop)
    plot0.view_init(20, 45)
    pyplot.savefig(plotname0 +'.pdf', format='pdf', dpi=2000)
    pyplot.close()
    # Plot 1: 2D Pareto Front Plots
    # -------------------------------------------------------------------------
    seaborn.set_style('darkgrid')
    objectives = [[0, 1], [0, 2], [1, 2]]
    for i, j in objectives:
        plotname1 = folder + 'ParetoFront' + str(i+1) + str(j+1) + '.eps'
        plot1 = pyplot.figure(dpi=2000)
        ax1 = plot1.add_subplot(111)
        for opal in range(nop):
            x = getcolumn(i, opal, data)
            y = getcolumn(j, opal, data)
            ax1.scatter(x, y, s=40, c=color_choices[opal], marker='o', label=methods[opal])
        ax1.set_xlabel(opcat[i])
        ax1.set_ylabel(opcat[j])
        ax1.legend(loc='upper right', frameon=True)
        pyplot.savefig(plotname1, format='eps')
        pyplot.close()
    plot_scatter_approxset(data, opcat, nop, folder, methods, color_choices,
                           ignore_method=ignore_method)


def plot_scatter_approxset(data, opcat, nop, folder, methods, color_choices,
                           ignore_method=False):
    # Plot 2: Scatter Matrix Plot
    plotname2 = folder + 'ParetoPlot'
    if ignore_method:
        scat = seaborn.PairGrid(data[0], vars=opcat)
        scat = scat.map_diag(pyplot.hist, facecolor=color_choices[0])
        scat = scat.map_offdiag(pyplot.scatter, color=color_choices[0], linewidths=1,
                                edgecolor="w", s=40)
    else:
        dataset = pandas.concat(data, keys=methods)
        scat = seaborn.PairGrid(dataset, vars=opcat, hue='Method',
                                palette=seaborn.color_palette(color_choices),
                                hue_kws={"marker": markers[:nop]})
        scat = scat.map_diag(pyplot.hist)
        scat = scat.map_offdiag(pyplot.scatter, linewidths=1, edgecolor="w", s=40)
    # Set the tick labels to be at a 45 degree angle for better fit
    for ax in scat.axes.flat:
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda xax, p: format(int(xax))))
        pyplot.setp(ax.get_xticklabels(), rotation=45)
    if not ignore_method:
        scat.add_legend(title=None, frameon=True)
        scat.fig.get_children()[-1].set_bbox_to_anchor((0.995, 0.925, 0, 0))
    pyplot.savefig(plotname2 + '.eps', format='eps', dpi=4000)
    pyplot.savefig(plotname2 + '.pdf', format='pdf', dpi=4000)
    pyplot.close()


def getcolumn(index, opal, data):
    # This function figures out which characteristic is being selected.
    if index == 0:
        column = data[opal]['No. of Bins']
    elif index == 1:
        column = data[opal]['Avg. Initial Bin Heat (W)']
    else:
        column = data[opal]['Max. Time to Move (s)']
    return column


def combineindicators(n, nop, runs, solsindex, datacollect, opcat, folder,
                      methods, wpareto=False):
    # This function combines the results given by getbinindicators(A, B)
    # into a simple histogram.
    print('Evaluating Performance.')
    df_data = pandas.concat(datacollect)
    df_list = []
    # Calculate binary-epsilon indicator
    df_ibivalues, df_epsilon = iepsilon(nop, runs, df_data, folder, methods)
    df_list.append(['Epsilon Values', df_ibivalues])
    df_list.append(['Epsilon Stats', df_epsilon])
    print(' - epsilon complete')
    # Calculate coverage indicator
    df_icovers, df_coverage = icoverage(nop, runs, df_data, folder, methods)
    df_list.append(['Coverage Values', df_icovers])
    df_list.append(['Coverage Stats', df_coverage])
    print(' - coverage complete')
    # Calculate diversity indicator
    df_spread = getspread(nop, runs, df_data, opcat, methods)
    makespreadplot(df_spread, folder)
    df_numsols = countsols(solsindex, runs, n, methods[:nop], df_data)
    df_list.append(['Number Solutions', df_numsols])
    df_list.append(['Spread Indicators', df_spread])
    print(' - diversity complete')
    if wpareto:
        df_pareto = pareto_compare(nop, runs, df_data, folder,
                                   methods, opcat)
        df_list.append(['Pareto Measures', df_pareto])
        print(' - comparison to Pareto Front complete')
    makeperformxlsx(df_list, folder)
    return df_numsols


def makeperformxlsx(df_list, folder):
    # Write data to PerformanceMetrics.xlsx
    writer = pandas.ExcelWriter(folder + 'PerformanceMetrics.xlsx')
    for f in range(len(df_list)):
        df_list[f][1].to_excel(writer, sheet_name=df_list[f][0])
    writer.save()


def iepsilon(nop, runs, df_data, folder, methods, wpareto=False):
    # This function controls the creation of the individual binary epsilon
    # indicators and their combination into overall indicators.
    # Reference: Zitzler, 2003
    df_ibivalues = makedf_binary(nop, runs, df_data, methods)
    epsilonmatrix = makebinarymatrix(nop, runs, df_ibivalues, folder, methods)
    # Convert overall matrix into Dataframe
    tuples = []
    for opal in range(nop):
        tuples.append((methods[opal], 'Average'))
        tuples.append((methods[opal], 'St. Dev.'))
    if wpareto:
        tuples.append((methods[nop], 'Average'))
        tuples.append((methods[nop], 'St. Dev.'))
    indexa = pandas.MultiIndex.from_tuples(tuples, names=['MethodA', ''])
    indexb = pandas.Index(methods, name='MethodB')
    df_epsilon = pandas.DataFrame(epsilonmatrix, index=indexb, columns=indexa)
    return df_ibivalues, df_epsilon


def makedf_binary(nop, runs, df_data, methods):
    df_ivalues = emptydf_indicators(nop, runs, methods)
    # Calculate binary epsilon indicators
    for a in range(nop):
        for b in range(a+1, nop):
            for ra in range(runs):
                data_a = df_data[(df_data['Method'] == methods[a]) &
                                 (df_data['Experiment'] == ra + 1)]
                for rb in range(runs):
                    data_b = df_data[(df_data['Method'] == methods[b]) &
                                     (df_data['Experiment'] == rb + 1)]
                    # A into B
                    i_ab = getbinindicators(data_a, data_b)
                    df_ivalues.set_value((methods[b], str(rb+1)),
                                         (methods[a], str(ra+1)), i_ab)
                    # B into A
                    i_ba = getbinindicators(data_b, data_a)
                    df_ivalues.set_value((methods[a], str(ra+1)),
                                         (methods[b], str(rb+1)), i_ba)
    return df_ivalues


def makebinarymatrix(nop, runs, df_ibivalues, folder, methods):
    # Open pyplot figure
    plotname = folder + 'Epsilon'
    figsize = 4 * 2, 3 * 2
    seaborn.set_context('paper', font_scale=1.25)
    ymin = round(df_ibivalues.values.min() - 0.05, 2)
    ymax = round(df_ibivalues.values.max() + 0.05, 2)
    fig = pyplot.figure(figsize=figsize)
    # Calculate overall matrix
    part = 0
    binarymatrix = np.ones((nop, 2 * nop))
    for a in range(nop):
        for b in range(a + 1, nop):
            # Find A into B
            df1 = gather_ivalues(a, b, runs, runs, df_ibivalues, methods)
            binarymatrix[b, 2 * a] = df1.mean().get_value('I_C', 0)
            binarymatrix[b, 2 * a + 1] = df1.std().get_value('I_C', 0)
            # Find B into A
            df2 = gather_ivalues(b, a, runs, runs, df_ibivalues, methods)
            binarymatrix[a, 2 * b] = df2.mean().get_value('I_C', 0)
            binarymatrix[a, 2 * b + 1] = df2.std().get_value('I_C', 0)
            # Plot values
            df_avb = pandas.concat([df1, df2])
            ax = fig.add_subplot(2, 2, part + 1, ylim=(ymin, ymax))
            ax = plotepsilon(ax, nop, df_avb, a, b, part)
            part += 1
    fig.tight_layout()
    pyplot.savefig(plotname + '.eps', format='eps', dpi=2000)
    pyplot.savefig(plotname + '.pdf', format='pdf')
    pyplot.close()
    return binarymatrix


def getbinindicators(opal_a, opal_b):
    # This function calculates the binary epsilon-indicators I_eta(A,B)
    # and I_eta(B,A) where A is the first method and B is the second method.
    eta12 = np.zeros((len(opal_a), len(opal_b)))
    for a in range(len(opal_a)):
        a_0 = opal_a.get_value(a, 0, takeable=True)
        a_1 = opal_a.get_value(a, 1, takeable=True)
        a_2 = opal_a.get_value(a, 2, takeable=True)
        for b in range(len(opal_b)):
            ratios = np.array([(a_0 / (opal_b.get_value(b, 0, takeable=True))),
                               (a_1 / (opal_b.get_value(b, 1, takeable=True))),
                               (a_2 / (opal_b.get_value(b, 2, takeable=True)))])
            eta12[a, b] = np.amax(ratios)
    eta2 = np.amin(eta12, axis=0)
    i_ab = np.amax(eta2)
    return i_ab


def plotepsilon(ax, nop, df_avb, a, b, part):
    # This function plots subplot #part
    plotcolors = [colors[a], colors[b], colors[nop]]
    seaborn.boxplot(x='Comparison', y='I_C', data=df_avb, ax=ax,
                    palette=plotcolors)
    ax.set_xlabel(figlabel[part])
    if part % 2 == 0:
        ax.set_ylabel('$I_\epsilon(A,B)$', labelpad=10)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels("")
    return ax


def icoverage(nop, runs, df_data, folder, methods, wpareto=False):
    # This module calculates the coverage indicator
    df_cover = makedf_cover(nop, runs, df_data, methods, wpareto=wpareto)
    coveragematrix = makecoveragematrix(nop, runs, df_cover, folder, methods)
    # Convert overall matrix into Dataframe
    tuples = []
    for opal in range(nop):
        tuples.append((methods[opal], 'Average'))
        tuples.append((methods[opal], 'St. Dev.'))
    if wpareto:
        tuples.append((methods[nop], 'Average'))
        tuples.append((methods[nop], 'St. Dev.'))
    indexa = pandas.MultiIndex.from_tuples(tuples, names=['MethodA', ''])
    indexb = pandas.Index(methods, name='MethodB')
    df_comcover = pandas.DataFrame(coveragematrix, index=indexb, columns=indexa)
    return df_cover, df_comcover


def makedf_cover(nop, runs, df_data, methods, wpareto=False):
    df_cover = emptydf_indicators(nop, runs, methods, with_pareto=wpareto)
    # Calculate coverage indicators
    for a in range(nop):
        for b in range(a+1, nop):
            for ra in range(runs):
                data_a = df_data[(df_data['Method'] == methods[a]) &
                                 (df_data['Experiment'] == ra + 1)]
                for rb in range(runs):
                    data_b = df_data[(df_data['Method'] == methods[b]) &
                                     (df_data['Experiment'] == rb + 1)]
                    # A into B
                    i_covab = calc_coverage(data_a, data_b)
                    df_cover.set_value((methods[b], str(rb+1)),
                                       (methods[a], str(ra+1)), i_covab)
                    # B into A
                    i_covba = calc_coverage(data_b, data_a)
                    df_cover.set_value((methods[a], str(ra+1)),
                                       (methods[b], str(rb+1)), i_covba)
    if wpareto:
        data_pareto = df_data[df_data['Method'] == methods[nop]]
        for a in range(nop):
            for ra in range(runs):
                data_a = df_data[(df_data['Method'] == methods[a]) &
                                 (df_data['Experiment'] == ra + 1)]
                # Pareto into A
                i_covpa = calc_coverage(data_pareto, data_a)
                df_cover.set_value((methods[a], str(ra + 1)),
                                   (methods[nop], str(1)), i_covpa)
                # A into Pareto
                i_covap = calc_coverage(data_a, data_pareto)
                df_cover.set_value((methods[nop], str(1)),
                                   (methods[a], str(ra + 1)), i_covap)
    return df_cover


def makecoveragematrix(nop, runs, df_cover, folder, methods):
    # Open pyplot figure
    plotname = folder + 'Coverage'
    figsize = 4 * 2, 3 * 2
    seaborn.set_context('paper', font_scale=1.25)
    ymin = max(round(df_cover.values.min() - 0.05, 2), 0.0)
    fig = pyplot.figure(figsize=figsize)
    # Calculate overall matrix
    part = 0
    coveragematrix = np.ones((nop, 2 * nop))
    for a in range(nop):
        for b in range(a + 1, nop):
            # Find A into B
            df1 = gather_ivalues(a, b, runs, runs, df_cover, methods)
            coveragematrix[b, 2 * a] = df1.mean().get_value('I_C', 0)
            coveragematrix[b, 2 * a + 1] = df1.std().get_value('I_C', 0)
            # Find B into A
            df2 = gather_ivalues(b, a, runs, runs, df_cover, methods)
            coveragematrix[a, 2 * b] = df2.mean().get_value('I_C', 0)
            coveragematrix[a, 2 * b + 1] = df2.std().get_value('I_C', 0)
            # Plot values
            df_avb = pandas.concat([df1, df2])
            ax = fig.add_subplot(2, 2, part + 1, ylim=(ymin, 1.0))
            ax = plotcoverage(ax, nop, df_avb, a, b, part)
            part += 1
    fig.tight_layout()
    pyplot.savefig(plotname + '.eps', format='eps', dpi=2000)
    pyplot.savefig(plotname + '.pdf', format='pdf')
    pyplot.close()
    return coveragematrix


def gather_ivalues(a, b, ra, rb, df_cover, methods):
    # This function filters df_cover for I_C values of algorithm A into B
    filter_metha = [(methods[a], str(r + 1)) for r in range(ra)]
    newshape = int(ra * rb)
    array_atob = df_cover.xs(methods[b], level='MethodB').as_matrix(columns=filter_metha)
    ic_values = np.reshape(array_atob, newshape)
    df = pandas.DataFrame(ic_values)
    df.columns = ['I_C']
    df['Comparison'] = '({0},{1})'.format(methods[a], methods[b])
    return df


def calc_coverage(data_a, data_b):
    # calculate coverage indicator of set a to set b
    n_bfront = len(data_b)      # fraction denominator
    n_bcovered = 0              # fraction numerator
    for mb in range(n_bfront):
        v = np.matrix([data_b.get_value(mb, 0, takeable=True),
                       data_b.get_value(mb, 1, takeable=True),
                       data_b.get_value(mb, 2, takeable=True)])
        v_covered = check_covered(data_a, v)
        if v_covered:
            n_bcovered += 1
        if data_a['Method'].any() == 'NewMethod':
            if not v_covered:
                print(v)
    i_covab = float(n_bcovered) / n_bfront
    return i_covab


def check_covered(data_a, v):
    # This function determines if any solution in set A covers vector v
    # belonging to set B
    for ma in range(len(data_a)):
        u = np.matrix([data_a.get_value(ma, 0, takeable=True),
                       data_a.get_value(ma, 1, takeable=True),
                       data_a.get_value(ma, 2, takeable=True)])
        if np.all(np.equal(u, v)) or dom(u, v):
            return True
    # If the solution made it all the way through set A, not covered
    return False


def plotcoverage(ax, nop, df_avb, a, b, part):
    # This function plots subplot #part
    plotcolors = [colors[a], colors[b], colors[nop]]
    seaborn.boxplot(x='Comparison', y='I_C', data=df_avb, ax=ax,
                    palette=plotcolors)
    ax.set_xlabel(figlabel[part])
    if part % 2 == 0:
        ax.set_ylabel('$I_C(A,B)$', labelpad=10)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels("")
    return ax


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


def emptydf_indicators(nop, runs, methods, with_pareto=False):
    # Create multilevel indices & empty dataframe
    tuples = []
    for a in range(nop):
        for r in range(runs):
            tuples.append((methods[a], str(r + 1)))
    if with_pareto:
        tuples.append((methods[nop], str(1)))
    indexa = pandas.MultiIndex.from_tuples(tuples, names=['MethodA', 'Experiment'])
    indexb = pandas.MultiIndex.from_tuples(tuples, names=['MethodB', 'Experiment'])
    df_indicators = pandas.DataFrame(np.ones((len(indexb), len(indexa))),
                                     index=indexb, columns=indexa)
    return df_indicators


def gettotalindicators(dfsolseries, solsindex, folder):
    # This function plots the number of solutions found for all levels.
    df_sols = pandas.concat(dfsolseries, axis=0)
    plt = seaborn.barplot(x=solsindex[0], y=solsindex[1], hue=solsindex[2], data=df_sols,
                          palette=seaborn.color_palette(colors), capsize=.2)
    plt.set(xlabel='Experiment Level', ylabel='Avg. # of Approximate Solutions Found')
    pyplot.savefig(folder + 'RangeofSolutions.eps', format='eps', dpi=2000)


def makeietaplots(a, b, ietas, folder, methods):
    # This function plots the distribution of I_etas of A into B
    plotname = folder + 'Indicator_' + methods[a] + '_into_' + methods[b] + '.eps'
    distab = np.zeros(len(ietas))
    for i in range(len(ietas)):
        distab[i] = ietas[i][b][a]
    seaborn.set_context('talk', font_scale=1.5)
    pyplot.figure(figsize=(8, 6))
    plot = seaborn.distplot(distab, kde=False, color=colors[a])
    plot.axes.set_title('({0},{1})'.format(methods[a], methods[b]), fontsize=28)
    start, end = plot.get_xlim()
    newend = round((end + 0.05), 1)
    plot.set_xlim(2 - newend, newend)
    pyplot.savefig(plotname, format='eps', dpi=2000)
    pyplot.close()


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


def getspread(nop, runs, df_data, opcat, methods, wpareto=False):
    # This module finds the spread indicator values for each algorithm
    # and then normalizes them to the largest value.
    # Calculate the spread of each front individually:
    spreadvals = np.zeros((runs, nop))
    for r in range(runs):
        for opal in range(nop):
            approxset = df_data[(df_data['Method'] == methods[opal]) &
                                (df_data['Experiment'] == r + 1)]
            spreadvals[r, opal] = calcspread(approxset, opcat)
    normfactor = np.max(spreadvals)
    if wpareto:
        data_pareto = df_data[df_data['Method'] == methods[nop]]
        pareto_spread = calcspread(data_pareto, opcat)
        normfactor = max(normfactor, pareto_spread)
    spread = spreadvals / normfactor
    # Combine spread indicators
    spreadindicators = []
    for r in range(runs):
        for opal in range(nop):
            spreadindicators.append([r + 1, spread[r, opal], methods[opal]])
    df_spread = pandas.DataFrame(spreadindicators)
    df_spread.columns = ['Run', 'Spread', 'Method']
    return df_spread


def calcspread(opaldata, opcat):
    # This module calculates the spread indicator from an approximate set
    dsum = 0
    for o in range(len(opcat)):
        maxval = opaldata.nlargest(1, opcat[o]).get_value(0, o, takeable=True)
        minval = opaldata.nsmallest(1, opcat[o]).get_value(0, o, takeable=True)
        dsum += (maxval - minval)**2
    dspread = sqrt(dsum)
    return dspread


def makespreadplot(df_spread, folder):
    # This module plots the spread indicators
    # x-axis: Algorithm
    # y-axis: Spread value
    kws = dict(s=70, linewidth=1, edgecolor="w")
    plotname = folder + 'SpreadIndicators'
    seaborn.set_context('paper', font_scale=1.25)
    plot = seaborn.boxplot(x='Method', y='Spread', data=df_spread,
                           palette=seaborn.color_palette(colors))
    plot.set(ylabel='Maximum Spread Indicator (Normalized)')
    # plot.set(yscale='log')
    plot.set(ylim=(0, 1))
    pyplot.savefig(plotname + '.eps', format='eps')
    pyplot.savefig(plotname + '.pdf', format='pdf')
    pyplot.close()


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


def countsols(solsindex, runs, levellabel, methods, df_data):
    # This function counts the number of solution generated by each algorithm
    # in each experiment.
    count_exp = df_data.groupby(['Experiment', 'Method']).count()
    solsdiversity = []
    for r in range(runs):
        for opal in range(len(methods)):
            mthd = methods[opal]
            nsols = count_exp.get_value((r+1, mthd), 'No. of Bins')
            solsdiversity.append([levellabel, nsols, mthd])
    df_numsols = pandas.DataFrame(solsdiversity)
    df_numsols.columns = solsindex
    return df_numsols


def pareto_compare(nop, runs, df_data, folder, methods, opcat):
    # This function evaluates each approximate set based on the found Pareto Front.
    # It calculates the absolute efficiency, the maximum distance, the average
    # distance, and the distance variability measures.
    indicators = []
    pareto_set = get_pareto_set(nop, runs, df_data, methods)
    paretoplot([pareto_set], opcat, 1, folder, ['Pareto Front'], [colors[nop]], ignore_method=True)
    # Normalize objectives to 100:
    max_ps = pareto_set.max(numeric_only=True)
    a = max_ps[0]
    b = max_ps[1]
    c = max_ps[2]
    lambda_theta = np.array([(100 / a), (100 / b), (100 / c)])
    for opal in range(nop):
        for r in range(runs):
            approxset = df_data[(df_data['Method'] == methods[opal]) &
                                (df_data['Experiment'] == r + 1)]
            # Calculate efficiency
            abs_efficiency = calc_efficiency(pareto_set, approxset)
            # Calculate individual distances
            d_values = get_dvalues(lambda_theta, pareto_set, approxset)
            # Calculate Max. Distance
            max_distance = np.amax(d_values)
            # Calculate Avg. Distance
            avg_distance = np.average(d_values)
            # Calculate Var. Distance
            var_distance = np.var(d_values, ddof=1)     # sample variance
            # Combine indicators
            indicators.append({
                'Run': r + 1,
                'Method': methods[opal],
                'Abs. Efficiency': abs_efficiency,
                'Max. Distance': max_distance,
                'Avg. Distance': avg_distance,
                'Var. Distance': var_distance,
                'n_f': d_values.shape[0],
                'Pooled St. Dev.': None
            })
        indicators = include_pooled_stdev(opal, runs, indicators, methods)
    df_pareto = pandas.DataFrame(indicators)
    plot_pareto(df_pareto, folder)
    return df_pareto


def include_pooled_stdev(opal, runs, indicators, methods):
    # This function calculates the pooled variance for method opal, adds the
    # pooled standard deviation to the indicators list, and returns it.
    # Reference: https://en.wikipedia.org/wiki/Pooled_variance
    poolvar_top = 0
    poolvar_bottom = 0
    for r in range(runs):
        n_f = indicators[-1 - r].get('n_f')
        poolvar_top += (n_f - 1) * indicators[-1 - r].get('Var. Distance')
        poolvar_bottom += n_f
    pooled_variance = poolvar_top / (poolvar_bottom - runs)
    pooled_stdev = sqrt(pooled_variance)
    indicators.append({
        'Run': None,
        'Method': methods[opal],
        'Abs. Efficiency': None,
        'Max. Distance': None,
        'Avg. Distance': None,
        'Var. Distance': None,
        'n_f': None,
        'Pooled St. Dev.': pooled_stdev
    })
    return indicators


def get_pareto_set(nop, runs, df_data, methods):
    # This function double checks the found Pareto set against the approximation
    # sets and makes necessary adjustments to return the best known pareto set.
    # Initialize from df_data
    pareto_set = df_data[(df_data['Method'] == 'Pareto')]
    for opal in range(nop):
        for r in range(runs):
            approxset = df_data[(df_data['Method'] == methods[opal]) &
                                (df_data['Experiment'] == r + 1)]
            # Double check that none of the solutions are nondominated
            for mb in range(approxset.shape[0]):
                v = np.matrix([approxset.get_value(mb, 0, takeable=True),
                               approxset.get_value(mb, 1, takeable=True),
                               approxset.get_value(mb, 2, takeable=True)])
                nondominated = nondominated_byset(v, pareto_set)
                if nondominated:
                    pareto_set = pareto_set.append(approxset.iloc[mb], ignore_index=True)
                    pareto_set = edit_paretoset_dataframe(v, pareto_set)
    return pareto_set


def nondominated_byset(v, seta):
    # This function determines if the objective vector v is nondominated by the
    # solutions in set A. Returns true or false.
    for ma in range(seta.shape[0]):
        u = np.matrix([seta.get_value(ma, 0, takeable=True),
                       seta.get_value(ma, 1, takeable=True),
                       seta.get_value(ma, 2, takeable=True)])
        if dom(u, v):
            return False
    # If v was not dominated by any solution in set A, it's nondominated too
    return True


def edit_paretoset_dataframe(v, pareto_set):
    # This function determines if any solution in the pareto_set is dominated
    # by vector v and removes it
    index_list = []
    for ma in range(pareto_set.shape[0]):
        u = np.matrix([pareto_set.get_value(ma, 0, takeable=True),
                       pareto_set.get_value(ma, 1, takeable=True),
                       pareto_set.get_value(ma, 2, takeable=True)])
        if dom(v, u):
            index_list.append(ma)
    pareto_set = pareto_set.drop(pareto_set.index[index_list])
    return pareto_set


def solution_in_paretofront(v, pareto_set):
    # This function verifies if v matches anything in pareto_set
    for ma in range(pareto_set.shape[0]):
        u = np.matrix([pareto_set.get_value(ma, 0, takeable=True),
                       pareto_set.get_value(ma, 1, takeable=True),
                       pareto_set.get_value(ma, 2, takeable=True)])
        if np.all(np.equal(u, v)):
            return True
    # If v was not found in pareto_set, return false
    return False


def calc_efficiency(pareto_set, approxset):
    # This function determines the absolute efficiency of an approximate set
    df_madeit = approxset[approxset.isin(pareto_set)].dropna()
    for mb in range(approxset.shape[0]):
        v = np.matrix([approxset.get_value(mb, 0, takeable=True),
                       approxset.get_value(mb, 1, takeable=True),
                       approxset.get_value(mb, 2, takeable=True)])
        inpareto = solution_in_paretofront(v, pareto_set)
        if inpareto:
            df_madeit = df_madeit.append(approxset.iloc[mb])
    num_madeit = df_madeit.shape[0]
    num_pareto = pareto_set.shape[0]
    abs_eff = num_madeit / num_pareto
    return abs_eff


def get_dvalues(lambda_theta, pareto_set, approxset):
    # This function determines how far each solution in approxset is from the
    # Pareto Front.
    d_values = np.zeros(approxset.shape[0], np.float)
    for mb in range(approxset.shape[0]):
        v = np.array([approxset.get_value(mb, 0, takeable=True),
                       approxset.get_value(mb, 1, takeable=True),
                       approxset.get_value(mb, 2, takeable=True)])
        d_values[mb] = get_distance_from_pareto(lambda_theta, v, pareto_set)
    return d_values


def get_distance_from_pareto(lambda_theta, v, pareto_set):
    # This function determines how far objective vector v is from the Pareto set
    all_distances = []
    for ma in range(pareto_set.shape[0]):
        u = np.array([pareto_set.get_value(ma, 0, takeable=True),
                       pareto_set.get_value(ma, 1, takeable=True),
                       pareto_set.get_value(ma, 2, takeable=True)])
        all_distances.append(calc_distance(lambda_theta, u, v))
    d_v = min(all_distances)
    return d_v


def calc_distance(lambda_theta, u, v):
    # This function calculates the Tchebycheff norm, with weight vector lambda.
    # It determines the distance between two objective vectors in the objective
    # space.
    diff = np.absolute(np.subtract(u, v))
    d = np.dot(lambda_theta, diff)
    return d


def plot_pareto(df_pareto, folder):
    # This module plots box plots for the absolute efficiency and the distance
    # measures.
    plot_absefficiency(df_pareto, folder)
    plot_distance_stdev(df_pareto, folder)
    plotname = folder + 'Distance_Measures'
    seaborn.set_context('paper', font_scale=1.25)
    fig, ax = pyplot.subplots(1, 2, sharey=True)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, w/2)
    labels = ['Avg. Distance from Pareto Front', 'Max. Distance from Pareto Front']
    for i in range(2):
        column_name = df_pareto.filter(regex='Distance').columns.values[i]
        seaborn.boxplot(x='Method', y=column_name, data=df_pareto,
                        palette=seaborn.color_palette(colors), ax=ax[i])
        ax[i].set_ylabel(labels[i])
    fig.tight_layout()
    pyplot.savefig(plotname + '.eps', format='eps')
    pyplot.savefig(plotname + '.pdf', format='pdf')
    pyplot.close()


def plot_absefficiency(df_pareto, folder):
    plotname = folder + 'Absolute_Efficiency'
    seaborn.set_context('paper', font_scale=1.25)
    plot = seaborn.boxplot(x='Method', y='Abs. Efficiency', data=df_pareto,
                           palette=seaborn.color_palette(colors))
    plot.set(ylabel='Absolute Efficiency')
    plot.set(ylim=(0, 1))
    pyplot.savefig(plotname + '.eps', format='eps')
    pyplot.savefig(plotname + '.pdf', format='pdf')
    pyplot.close()


def plot_distance_stdev(df_pareto, folder):
    plotname = folder + 'Distance_StdDeviation'
    seaborn.set_context('paper', font_scale=1.25)
    plot = seaborn.barplot(x='Method', y='Pooled St. Dev.', data=df_pareto,
                           palette=seaborn.color_palette(colors))
    plot.set(ylabel='Pooled Standard Deviation')
    avg_max = df_pareto['Avg. Distance'].max()
    plot.set(ylim=(0, ceil(avg_max)))
    pyplot.savefig(plotname + '.eps', format='eps')
    pyplot.savefig(plotname + '.pdf', format='pdf')
    pyplot.close()


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


def geth5xyts(folder):
    # This function imports the h5 file to retrieve the x and y matrices
    hfile = folder + 'xymatrices.h5'     # File name
    xgrp = 'xmatrices'          # Group name: x matrices
    ygrp = 'yarrays'            # Group name: y arrays
    tgrp = 'tfills'             # Group name: tfill matrices
    # Verify that path exists and is accessible
    if not os.path.isfile(hfile):
        print('File not found: {0}'.format(hfile))
    if not os.access(hfile, os.R_OK):
        print('File not readable: {0}'.format(hfile))
    print('     -  importing matrices')
    xitems = []
    yitems = []
    titems = []
    with h5py.File(hfile, 'r') as h5f:
        gx = h5f.get(xgrp)
        gy = h5f.get(ygrp)
        gt = h5f.get(tgrp)
        solids = keys(gx)
        for m in range(len(solids)):
            datax = gx.get(solids[m])
            xitems.append(np.array(datax))
            datay = gy.get(solids[m])
            yitems.append(np.array(datay))
            datat = gt.get(solids[m])
            titems.append(np.array(datat))
    return solids, xitems, yitems, titems


def keys(f):
    # Returns a list of h5 keys
    return [key for key in f.keys()]


def listfolders(r, lvl, nop, folders, levels):
    # This function combines the folder names together.
    runfolds = []
    for opal in range(nop):
        drcty = folders[opal] + levels[lvl]
        runfolds.append(drcty + 'Experiment{0:02d}/'.format(r+1))
    runfolds.append(folders[-1] + levels[lvl] + 'Experiment{0:02d}/'.format(r+1))
    pathname = os.path.dirname(runfolds[nop])
    if not glob(pathname):
        os.mkdir(pathname)
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

#------------------------------------------------------------------------------
#
#           Box Feature Functions
#
#------------------------------------------------------------------------------


def getbininfo(cookies, nop, folders, methods):
    # This module recreates the features of the bins in each solution
    info_file = folders[-1] + 'IndividualBinInfo.xlsx'
    writer = pandas.ExcelWriter(info_file, engine='openpyxl')
    for opal in range(nop):
        solids, xitems, yitems, titems = geth5xyts(folders[opal])
        df_heatlist = []
        df_readylist = []
        for m in range(len(solids)):
            # Get individual box initial heat levels
            q0bins = box_heat(cookies, xitems[m], yitems[m], titems[m])
            df_q0m = pandas.DataFrame(q0bins)
            df_heatlist.append(df_q0m)
            # Get individual box ready times
        df_heat = pandas.concat(df_heatlist, ignore_index=True, axis=1)
        df_heat.columns = solids
        df_heat.to_excel(writer, sheet_name=methods[opal]+'_initheat')
        writer.save()


def box_heat(cookies, x, y, tfill):
    # This function calculates the average initial heat of the boxes
    q0bins = np.zeros(len(cookies), dtype=np.float)
    for i in range(len(cookies)):
        if y[i] == 1:
            q0bins[i] = getinitialboxheat(cookies, i, x, tfill[i])
    return q0bins


def getinitialboxheat(cookies, i, x, tfilli):
    # This function calculates the initial heat in box i (W)
    h = 8.0                 # Heat Transfer Coefficient, air [W/m2C]
    tempamb = 298.0         # ambient air temp. [K]
    sa = cookies.get(0).surfarea
    boxheat = 0
    for j in range(len(cookies)):
        if x[i, j] == 1:
            # Get temperature of cookie at time tfill[i]
            tempcookie = cookies.get(j).gettemp(tfilli, tempamb, h)
            # Calculate convective heat from cookie
            heatfromcookie = h * sa * (tempcookie - tempamb)
            boxheat += heatfromcookie
    return boxheat


#------------------------------------------------------------------------------
#
#           Dynamic Constraint Functions
#
#------------------------------------------------------------------------------


def checkforviolations(n, folders, cookies, methods):
    # This function checks an approximation set for constraint violations
    boxcap = 8                  # Capacity of box set in MOCookieProblem
    coolrack = 15               # Cooling rack capacity
    fillcap = 2                 # Limit on number of boxes filled per period
    nbatches = 4                # Number of batches baked
    # boxcap = 24                 # Capacity of box set in MOCookieProblem
    # coolrack = 300              # Cooling rack capacity
    # fillcap = 8                 # Limit on number of boxes filled per period
    # nbatches = 10               # Number of batches baked
    tbatch = 600                # Time to cook one batch of cookies
    print('Checking solutions for feasibility.')
    for opal in range(len(folders) - 1):
        print(' - checking method ' + methods[opal])
        solids, xitems, yitems, titems = geth5xyts(folders[opal])
        for m in range(len(solids)):
            noreplacement(n, solids[m], xitems[m])
            boxcapacity(n, solids[m], xitems[m], boxcap)
            timeconstraint(n, solids[m], xitems[m], titems[m], cookies, tbatch)
            rackcapacity(n, xitems[m], titems[m], cookies, coolrack, tbatch)
            period_fill_limit(solids[m], yitems[m], titems[m],
                              fillcap, tbatch, nbatches)


def noreplacement(n, solid, x):
    # This function ensures that x respects the "no replacement constraint
    # Return false if an error is detected; the code should never violate
    # this, so if error output is present, algorithm has a bug.
    itemspresent = np.sum(x, axis=0)
    for j in range(n):
        if itemspresent[j] > 1:
            print('     Solution', solid, 'has a physicality error: item', j)
            return False
    return True


def boxcapacity(n, solid, x, boxcap):
    # This function ensures that no box is filled beyond capacity
    # Return false if an error is detected; the code should never violate
    # this, so if error output is present, algorithm has a bug.
    boxitems = np.sum(x, axis=1)
    for i in range(n):
        if boxitems[i] > boxcap:
            print('     Error: Solution', solid, 'has filled bin', i, 'beyond capacty.')
            return False
    return True


def timeconstraint(n, solid, x, tfill, cookies, tbatch):
    # This function ensures that no cookie is put in a box before it
    # even gets out of the oven. Returns a list of (i,j) tuples for each
    # violation.
    violations = []
    for i in range(n):
        for j in range(n):
            if x[i, j] == 1:
                baked = cookiedonebaking(cookies.get(j), tfill[i], tbatch)
                if baked is False:
                    print('     Error: Solution', solid, 'has packed cookie', j,
                          'in box', i, 'before it finished baking.')
                    violations.append((i, j))
    return violations


def rackcapacity(n, x, tfill, cookies, coolrack, tbatch):
    # This function checks that the cooling rack is never be filled
    # beyond capacity and collects a list of violations as (i,j) tuples.
    tints = timeintervals(n, tfill, tbatch)
    violations = []
    for t in tints:
        # Cookies from boxes filled after t might be on rack
        timecheckindices = np.where(tfill > t)
        cookiesonrack = []
        for i in timecheckindices[0]:
            for j in range(n):
                if x[i, j] == 1:
                    onrack = rackij(t, tfill[i], cookies.get(j), tbatch)
                    if onrack == 1:
                        cookiesonrack.append((i, j))
        if len(cookiesonrack) > coolrack:
            violations.append([t, cookiesonrack])
    return violations


def cookiedonebaking(cookie, t, tbatch):
    # This function checks if cookie j is out of the oven by time t
    # Return True if cookie is out of the oven, otherwise return False
    bk = cookie.getbatch()
    if bk * tbatch > t:
        return False
    else:
        return True


def timeintervals(n, tfill, tbatch):
    # This module collects a list of time intervals at which rackcapacity()
    # should check.
    times = np.sort(tfill)
    timeintervals = [tbatch]
    for i in range(n):
        # Want at least 300 seconds in between intervals
        if times[i] >= timeintervals[-1] + 300:
            timeintervals.append(times[i])
    return timeintervals


def rackij(t, tfill, cookie, tbatch):
    # This function determines if cookie is present on the cooling rack
    # at time t. It returns 1 for present and 0 for absent.
    t0 = cookie.batch * tbatch
    if t0 <= t < tfill:
        return 1
    else:
        return 0


def period_fill_limit(solid, y, tfill, fillcap, tbatch, nbatches):
    # This function checks that the number of casks filled in each time
    # period does not exceed the limit.
    t_t = get_filltime_periods(tfill, tbatch, nbatches)
    res_fill = make_filltime_residuals(t_t, tfill, y, fillcap)
    # Check each time period
    violations = []
    for t in range(len(t_t) - 1):
        if res_fill[t] < 0:
            print('     Error: Solution {0} has filled too many bins in the time '
                  'period starting at {1} seconds.'.format(solid, t_t[t]))
            violations.append(t_t[t])
    return violations


def make_filltime_residuals(t_t, tfill, y, fillcap):
    # This module forms the residual matrix for the fill period limit.
    m = np.sum(y)
    res_fill = []
    for t in range(len(t_t) - 1):
        p_t = [i for i in range(m) if t_t[t] <= tfill[i] < t_t[t + 1]]
        res_fill.append(fillcap - len(p_t))
    return res_fill


def get_filltime_periods(tfill, tbatch, nbatches):
    # Get time periods that define the fill limit
    t_end = max(np.amax(tfill), tbatch * nbatches)
    n_period = int(2 * (t_end - tbatch) // tbatch) + 2
    t_t = [tbatch * (1.0 + t / 2.0) for t in range(n_period)]
    return t_t


if __name__ == '__main__':
    main()
