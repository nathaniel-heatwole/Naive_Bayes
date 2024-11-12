# NAIVE_BAYES.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Uses a naive Bayes classifier to predict level (discrete) of body mass index (bmi = weight / height^2)
# Training data: empirical health-related data for 741 persons (from https://www.kaggle.com/datasets/rukenmissonnier/age-weight-height-bmi-analysis)

import time
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from colorama import Fore, Style
from sklearn.naive_bayes import GaussianNB

time0 = time.time()
ver = ''  # version (empty or integer)

topic = 'Naive Bayes'
topic_underscore = topic.replace(' ','_')

#--------------#
#  PARAMETERS  #
#--------------#

y_var = 'bmi group 4 num'
covariates = ['wgt', 'hgt', 'age']

var_units = {'bmi':'kg/m^2', 'wgt':'kg', 'hgt':'m', 'age':'yrs'}

bmi_labels_6 = ['underweight', 'normal weight', 'overweight', 'obese class 1', 'obese class 2', 'obese class 3']
bmi_labels_4 = ['underweight', 'normal weight', 'overweight', 'obese']
class_labels = bmi_labels_4

#-----------------#
#  DATA CLEANING  #
#-----------------#

# bmi levels dictionary
bmi_labels_dict = {}
for g in range(len(class_labels)):
    bmi_labels_dict[class_labels[g]] = g

# import training data
bmi = pd.read_csv('bmi.csv')
total_obs = len(bmi)

# column names -> lowercase
for col in bmi.columns:
    bmi[col.lower()] = bmi[col]
    bmi.drop(columns=col, axis=1, inplace=True)
del col

# rename columns
bmi.rename(columns={'bmiclass':'bmi group 6', 'weight':'wgt', 'height':'hgt'}, inplace=True) 

# bmi levels (discrete)
bmi['bmi group 6'] = [bmi['bmi group 6'][i].lower() for i in bmi.index]  # converts column values to lowercase
bmi['bmi group 4'] = bmi['bmi group 6'].apply(lambda x: 'obese' if x[0:len('obese')] == 'obese' else x)  # 4-level bmi (collapses all 'obese' into one level)
bmi['bmi group 4 num'] = bmi['bmi group 4'].apply(lambda x: bmi_labels_dict.get(x))

# missing values check
if bmi.isna().sum().sum() > 0:
    print(Fore.RED + '\033[1m' + '\n' + 'ERROR: TRAINING DATA CONTAINS MISSING VALUES' + '\n' + Style.RESET_ALL)

#----------------#
#  FROM SCRATCH  #
#----------------#

# model data
y = bmi[y_var]
x = bmi[covariates]

total_groups = len(y.unique())

# initialize
pct_var_names = []
mean_var_names = []
sigma_var_names = []
for cov in covariates:
    pct_var_names.append(cov + ' pct')
    mean_var_names.append(cov + ' avg')
    sigma_var_names.append(cov + ' std')

# actual counts (group-level)
counts = pd.DataFrame()
cnts = bmi.groupby(y_var)[covariates].count()
counts['cnt'] = cnts.iloc[:,0]  # only need the first column (assumes the training data contains no missing values)
del cnts

# initial group shares (overall - without regard to the values of the covariates)
percents = pd.DataFrame()
pcts = bmi.groupby(y_var)[covariates].count() / total_obs
percents['pct'] = pcts.iloc[:,0]  # only need the first column (assumes the training data contains no missing values)
del pcts

# bmi stats (group-level)
bmi_means = pd.DataFrame()
bmi_sigmas = pd.DataFrame()
bmi_stats = pd.DataFrame()
bmi_means['bmi avg'] = bmi.groupby(y_var)['bmi'].mean()
bmi_sigmas['bmi std'] = bmi.groupby(y_var)['bmi'].std()
bmi_stats = pd.concat([bmi_means, bmi_sigmas], axis=1)

# covariates stats (group-level (mean, sigma) (these are the maximum likelihood paramater estimates for a normal distribution)
means_covars = pd.DataFrame()
sigmas_covars = pd.DataFrame()
means_covars[mean_var_names] = bmi.groupby(y_var)[covariates].mean()
sigmas_covars[sigma_var_names] = bmi.groupby(y_var)[covariates].std()

# normal pdf (probability density function) values for each covariate-level combination (hence, Gaussian naive Bayes)
dists = pd.DataFrame()
for cov in covariates:
    for g in range(total_groups):
        dists['pdf ' + cov + ' ' + str(g)] = stats.norm.pdf(bmi[cov], loc=means_covars.at[g, cov + ' avg'], scale=sigmas_covars.at[g, cov + ' std'])

# product of pdfs (by group) and sum (over all groups)
dists['pdf sum'] = 0  # initialize (additive quantity)
for g in range(total_groups):    
    dists['pdf ' + str(g)] = percents.at[g, 'pct']  # initialize to be percent of obs in training data (multiplicative quantity)
    for cov in covariates:
        mult = dists['pdf ' + cov + ' ' + str(g)]
        dists['pdf ' + str(g)] *= mult  # multiplies on probability (pdf) for this covariate-level combination (multiplication assumes independence)
    dists['pdf sum'] += dists['pdf ' + str(g)]  # adds this probability (pdf) to the running total probability
del mult

# renormalize probabilities to sum to one (rowwise)
preds = pd.DataFrame()
for g in range(total_groups):
    preds['prob ' + str(g)] = dists['pdf ' + str(g)] / dists['pdf sum']

# predicted group
for g in range(total_groups):
    preds.loc[preds['prob ' + str(g)] == preds.max(axis=1), 'pred class'] = g  # rowwise maximum
preds[y_var] = y
preds['correct'] = [int(y[i] == preds['pred class'][i]) for i in preds.index]

#-----------#
#  SKLEARN  #
#-----------#

y_resized = np.resize(y, (len(y), ))

# initialize variable list
gnb_prob_var_names = []
for g in range(total_groups):
    gnb_prob_var_names.append('prob ' + str(g) + ' sklearn')
    
# fit classifier
gnb = GaussianNB(priors=None, var_smoothing=0)
gnb_fit = gnb.fit(x, y_resized)

# make predictions
preds[gnb_prob_var_names] = gnb_fit.predict_proba(x)
preds['pred class sklearn'] = gnb_fit.predict(x)
preds['methods align'] = [int(preds['pred class'][i] == preds['pred class sklearn'][i]) for i in preds.index]

# assess model fit
accuracy = sum(preds['correct']) / total_obs
accuracy_sklearn = gnb_fit.score(x, y_resized)

#-------------------#
#  SUMMARY RESULTS  #
#-------------------#

# lists (variables, units)
stats_vars = []
stats_units = []
for var in itertools.chain(['bmi'], covariates):
    stats_vars.append(var + ' avg')
    stats_vars.append(var + ' std')
    for i in [0, 1]:  # one for avg, and the other for std (same units)
        stats_units.append(var_units[var])

# training stats (merged)
training_stats = pd.concat([bmi_stats, means_covars, sigmas_covars], axis=1)
training_stats = training_stats[stats_vars]
training_stats = round(training_stats, 2)
training_stats.loc[len(training_stats)] = stats_units
training_stats.index = class_labels + ['(units)']

# model results
summary = pd.DataFrame()
summary['# actual'] = counts
summary['% actual'] = round(100 * percents, 2)
summary['% from scratch'] = 100 * preds.groupby('pred class')['pred class'].count() / total_obs
summary['% sklearn'] = 100 * preds.groupby('pred class sklearn')['pred class sklearn'].count() / total_obs
summary.loc[len(summary)] = ['', '', accuracy, accuracy_sklearn]
summary = round(summary, 2)
summary.index = class_labels + ['(accuracy)']

#----------#
#  EXPORT  #
#----------#

# functions
def console_print(subtitle, df, first_line, title_top):
    if first_line == 1:
        print(Fore.GREEN + '\033[1m' + '\n' + title_top + Style.RESET_ALL)
    print(Fore.GREEN + '\033[1m' + '\n' + subtitle + Style.RESET_ALL)
    print(df)
def txt_export(subtitle, df, f, first_line, title_top):
    if first_line == 1:
        print(title_top, file=f)
    print('\n' + subtitle, file=f)
    print(df, file=f)

title_top = topic.upper() + ' SUMMARY'
dfs = [training_stats, summary]
df_labels = ['training stats', 'model summary']

# export summary (console, txt)
with open(topic_underscore + '_summary' + ver + '.txt', 'w') as f:
    for out in ['console', 'txt']:
        first_line = 1
        for d in range(len(dfs)):
            subtitle = df_labels[d].upper()
            if out == 'console':
                console_print(subtitle, dfs[d], first_line, title_top)  # console print
            elif out == 'txt':
                txt_export(subtitle, dfs[d], f, first_line, title_top)  # txt file
            first_line = 0
del f, title_top, subtitle

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0, cov, g


