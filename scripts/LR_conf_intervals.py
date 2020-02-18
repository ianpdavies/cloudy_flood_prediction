# -*- coding: utf-8 -*-
# @Author: roman
# @Date:   2016-05-09 19:59:31
# @Last Modified by:   roman
# @Last Modified time: 2016-05-09 21:08:37
# From: https://gist.github.com/lqdc/1ea1682ad1214956d95904ebde3134a5

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import twenty_newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


CRIT_VALS = {99: 2.58, 95: 1.96, 90: 1.64}


def decision(coefs, X, intercept):
    return np.dot(X, coefs) + intercept


def get_se(X, y, clf):
    """StdErr per variable estimation.

    https://en.wikipedia.org/wiki/Ordinary_least_squares
    """
    MSE = np.mean((y - clf.predict(X).T)**2)
    # numerically unstable below with openblas if rcond is less than that
    var_est = MSE * np.diag(np.linalg.pinv(np.dot(X.T, X), rcond=1e-10))
    SE_est = np.sqrt(var_est)
    return SE_est


def get_probs(clf, X, SE_est, z=1.96):
    """Estimate CI given data, StdErrors and model."""
    coefs = np.ravel(clf.coef_)
    upper = coefs + (z * SE_est)
    lower = coefs - (z * SE_est)

    prob = 1. / (1. + np.exp(-decision(coefs, X, clf.intercept_)))
    upper_prob = 1. / (1. + np.exp(-decision(upper, X, clf.intercept_)))
    lower_prob = 1. / (1. + np.exp(-decision(lower, X, clf.intercept_)))

    stacked = np.vstack((lower_prob, upper_prob))
    up = np.max(stacked, axis=0)
    lo = np.min(stacked, axis=0)
    return prob, up, lo


def evaluate_using_heuristic(probs, upper, lower, target):
    correct_cnt = 0
    wrong_cnt = 0
    unsure_cnt = 0
    correct_unsure = 0
    wrong_unsure = 0

    for prob, up, lo, y in zip(probs, upper, lower, target):
        if prob >= 0.5 and lo >= 0.5:
            if y == 1:
                correct_cnt += 1
            elif y == 0:
                wrong_cnt += 1
        elif prob < 0.5 and up < 0.5:
            if y == 1:
                wrong_cnt += 1
            elif y == 0:
                correct_cnt += 1
        elif prob >= 0.5 > lo:
            unsure_cnt += 1
            if y == 1:
                correct_unsure += 1
            else:
                wrong_unsure += 1

        elif prob < 0.5 <= up:
            unsure_cnt += 1
            if y == 1:
                wrong_unsure += 1
            else:
                correct_unsure += 1

        else:
            raise ValueError(
                'Prob: %s, Upper: %s, Lower: %s, Target: %s' % (
                    prob, up, lo, y))

    print('Accuracy on sure: %s' % (correct_cnt / (correct_cnt + wrong_cnt),))
    print('Accuracy on unsure: %s' % (
        correct_unsure / (correct_unsure + wrong_unsure)))
    print('Number unsure:', unsure_cnt)
    print('Number sure: %s' % (correct_cnt + wrong_cnt,))


def check_ci(crit_val=95):
    try:
        z_score = CRIT_VALS[crit_val]
    except KeyError:
        print('Provide a value one of %s' % list(CRIT_VALS.keys()))
        return

    print('Using threshold %d%%' % crit_val)
    lr = LogisticRegression(C=1, dual=False, solver='lbfgs', max_iter=1000)
    svm = LinearSVC(C=0.3, max_iter=1000)
    sel = SelectFromModel(svm, prefit=False)
    sc = StandardScaler()
    vect = TfidfVectorizer(sublinear_tf=True, stop_words='english')

    categ = ['alt.atheism', 'talk.religion.misc']
    res = twenty_newsgroups.fetch_20newsgroups(categories=categ)
    data, y = res['data'], res['target']

    data_trn_val, data_tst, y_trn_val, y_tst = train_test_split(
        data, y, test_size=0.2, random_state=42)
    data_trn, data_val, y_trn, y_val = train_test_split(
        data_trn_val, y_trn_val, test_size=0.5, random_state=42)

    X_trn = sc.fit_transform(
        sel.fit_transform(vect.fit_transform(data_trn), y_trn).todense())
    X_val = sc.transform(sel.transform(vect.transform(data_val)).todense())
    X_tst = sc.transform(sel.transform(vect.transform(data_tst)).todense())

    lr.fit(X_trn, y_trn)

    # could be estimated from x-validation instead
    SE_est = get_se(X_val, y_val, lr)
    prob, up, lo = get_probs(lr, X_tst, SE_est, z=z_score)
    # print(up - lo)
    print('Accuracy normally:', lr.score(X_tst, y_tst))
    # 90.6%
    evaluate_using_heuristic(prob, up, lo, y_tst)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate CI for Logistic Regression')
    parser.add_argument('-t', '--threshold', type=int, default=95)
    args = parser.parse_args()
    check_ci(args.threshold)
