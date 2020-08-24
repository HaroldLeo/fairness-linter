import pandas as pd
from statistics import mean
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def fairness(data, label, pred, priv, unpriv, verbosity=1):
    if not isinstance(data, pd.DataFrame):
        print('ERROR: variable type of data must be pandas dataframe')
        return
    if not isinstance(label, str):
        print('ERROR: variable type of label must be string')
        return
    if not isinstance(pred, str):
        print('ERROR: variable type of pred must be string')
        return
    if not isinstance(priv, list):
        print('ERROR: variable type of priv must be list')
        return
    if not isinstance(unpriv, list):
        print('ERROR: variable type of unpriv must be list')
        return
    if not isinstance(verbosity, int):
        print('ERROR: variable type of verbosity must be int')
        return
    if len(data) == 0:
        print('ERROR: data is empty')
        return
    if len(priv) == 0:
        print('ERROR: pred is empty')
        return
    if len(data) == 0:
        print('ERROR: data is empty')
        return
    if not isinstance(priv[0], str):
        print('ERROR: variable type of elements in priv must be str')
        return
    if not isinstance(unpriv[0], str):
        print('ERROR: variable type of elements in unpriv must be str')
        return

    df = data.copy()

    fpr = []
    fnr = []
    priv_df = pd.DataFrame()
    unpriv_df = pd.DataFrame()
    temp = pd.DataFrame()
    sens = priv+unpriv
        
    for col1 in priv:
        temp = df.loc[df[col1] == 1]
        tn, fp, fn, tp = confusion_matrix(temp[label], temp[pred]).ravel()
        fpr.append(fp/(fp+tn))
        fnr.append(fn/(fn+tp))
        priv_df = priv_df.append(temp, ignore_index=True)

    if len(priv_df) == 0:
        print('ERROR: there is no data with given privileged columns')
        return

    for col2 in unpriv:
        temp = df.loc[df[col2] == 1]
        tn, fp, fn, tp = confusion_matrix(temp[label], temp[pred]).ravel()
        fpr.append(fp/(fp+tn))
        fnr.append(fn/(fn+tp))
        unpriv_df = unpriv_df.append(temp, ignore_index=True)

    if len(unpriv_df) == 0:
        print('ERROR: there is no data with given privileged columns')
        return

    fpr_max = max(fpr)
    fpr_max_col = sens[fpr.index(fpr_max)]
    fpr_min = min(fpr)
    fpr_min_col = sens[fpr.index(fpr_min)]
    fpr_mean = mean(fpr)
    fnr_max = max(fnr)
    fnr_max_col = sens[fnr.index(fnr_max)]
    fnr_min = min(fnr)
    fnr_min_col = sens[fnr.index(fnr_min)]
    fnr_mean = mean(fnr)
    
    priv_tn, priv_fp, priv_fn, priv_tp = confusion_matrix(priv_df[label], 
                                                          priv_df[pred]).ravel()
    priv_tpr = priv_tp/(priv_tp+priv_fn)
    priv_fpr = priv_fp/(priv_fp+priv_tn)

    unpriv_tn, unpriv_fp, unpriv_fn, unpriv_tp = confusion_matrix(unpriv_df[label], 
                                            unpriv_df[pred]).ravel()
    unpriv_tpr = unpriv_tp/(unpriv_tp+unpriv_fn)
    unpriv_fpr = unpriv_fp/(unpriv_fp+unpriv_tn)

    eod = unpriv_tpr - priv_tpr
    aod = ((unpriv_fpr - priv_fpr) + (unpriv_tpr - priv_tpr)) / 2

    priv_prob = len(priv_df.loc[(priv_df[pred] == 1)])/len(priv_df)
    unpriv_prob = len(unpriv_df.loc[(unpriv_df[pred] == 1)])/len(unpriv_df)
    di = unpriv_prob/priv_prob
    
    if verbosity >= 1:
        print('\n------------------------------Fairness tests results------------------------------\n')
        print('In this model:')
        print('- %s has the highest false positive rate at %f'%(fpr_max_col, fpr_max))
        print('- %s has the lowest false positive rate at %f'%(fpr_min_col, fpr_min))
        print('- %s has the highest false negative rate at %f'%(fnr_max_col, fnr_max))
        print('- %s has the lowest false negative rate at %f'%(fnr_min_col, fnr_min))
        print('- The mean false positive rate is %f'%fpr_mean)
        print('- The mean false negative rate is %f'%fnr_mean)
        if verbosity >= 2:
            N = len(sens)
            ind = np.arange(N) 
            width = 0.35
            plt.bar(ind, fpr, width, label='False Positive Rate')
            plt.bar(ind + width, fnr, width, label='False Negative Rate')
            plt.ylabel('Rate')
            plt.title('False Positive and False Negative Rate')
            plt.xticks(ind + width / 2, sens)
            plt.legend(loc='best')
            plt.show()
            if verbosity >= 3:
                df1 = pd.DataFrame([fpr, fnr], index=['FPR', 'FNR'], columns=priv+unpriv)
                print(df1)
        
        print('\n------------------------------Equal Opportunity Difference------------------------------\n')
        if eod < -0.1: 
            print('Based on the equal opportunity difference, this model implies higher benefit for the privileged group')
        if eod > 0.1:
            print('Based on the equal opportunity difference, this model implies higher benefit for the unprivileged group')
        if verbosity >= 2:
            fig = plt.figure()
            ax = fig.add_axes([0,0,0.8,0.8])
            ax.set_ylim([-1, 1])
            ax.bar([''], [eod], width=0.5)
            ax.grid(color='#808080', linestyle='--', linewidth=1, axis='y', alpha=0.7)
            plt.title('Equal Opportunity Difference')
            if eod < 0:
                ax.text(-0.02, eod - 0.1, str(round(eod, 2)))
            else:
                ax.text(-0.02, eod + 0.1, str(round(eod, 2)))
            plt.show()
            print('Fairness for the equal opportunty difference metric is between -0.1 and 0.1 with the ideal value at 0')
        
        print('\n------------------------------Average Odds Difference------------------------------\n')
        if aod < -0.1:
            print('Based on the average odds difference, this model implies higher benefit for the privileged group')
        if aod > 0.1:
            print('Based on the average odds difference, this model implies higher benefit for the unprivileged group')
        if verbosity >= 2:
            fig = plt.figure()
            ax = fig.add_axes([0,0,0.8,0.8])
            ax.set_ylim([-1, 1])
            ax.bar([''], [aod], width=0.5)
            ax.grid(color='#808080', linestyle='--', linewidth=1, axis='y', alpha=0.7)
            plt.title('Average Odds Difference')
            if aod < 0:
                ax.text(-0.02, aod - 0.1, str(round(aod, 2)))
            else:
                ax.text(-0.02, aod + 0.1, str(round(aod, 2)))
            plt.show()
            print('Fairness for the average odds difference metric is between -0.1 and 0.1 with the ideal value at 0')
            if verbosity >= 3:
                df2 = pd.DataFrame({'Priviledged': [priv_tpr, priv_fpr], 'Unpriviledged': [unpriv_tpr, unpriv_fpr]}, index=['TPR', 'FPR'])
                print('')
                print(df2)
        
        print('\n------------------------------Disparate Impact------------------------------\n')
        if di < 0.8:
            print('Based on the disparate impact, this model implies higher benefit for the privileged group')
        if di > 1.2:
            print('Based on the disparate impact, this model implies higher benefit for the unprivileged group')
        if verbosity >= 2:
            fig = plt.figure()
            ax = fig.add_axes([0,0,0.8,0.8])
            if di < 2:
                ax.set_ylim([0, 2])
            else: 
                ax.set_ylim([0, round(di+0.5)])
            ax.bar([''], [di], width=0.5)
            ax.grid(color='#808080', linestyle='--', linewidth=1, axis='y', alpha=0.7)
            plt.title('Disparate Impact')
            ax.text(-0.02, di + 0.1, str(round(di, 2)))
            plt.show()
            print('Fairness for the disparate impact metric is between 0.8 and 1.2 with the ideal value at 1')
            if verbosity >= 3:
                df3 = pd.DataFrame({'Priviledged': [priv_prob], 'Unpriviledged': [unpriv_prob]}, index=['Probability of predicted value = 1'])
                print('')
                print(df3)
    return

def intersectionality(data, label, pred, priv, unpriv, verbosity=1):
    if not isinstance(data, pd.DataFrame):
        print('ERROR: variable type of data must be pandas dataframe')
        return
    if not isinstance(label, str):
        print('ERROR: variable type of label must be string')
        return
    if not isinstance(pred, str):
        print('ERROR: variable type of pred must be string')
        return
    if not isinstance(priv, list):
        print('ERROR: variable type of priv must be list')
        return
    if not isinstance(unpriv, list):
        print('ERROR: variable type of unpriv must be list')
        return
    if not isinstance(verbosity, int):
        print('ERROR: variable type of verbosity must be int')
        return
    if len(data) == 0:
        print('ERROR: data is empty')
        return
    if len(priv) == 0:
        print('ERROR: pred is empty')
        return
    if len(data) == 0:
        print('ERROR: data is empty')
        return
    if not isinstance(priv[0], str):
        print('ERROR: variable type of elements in priv must be str')
        return
    if not isinstance(unpriv[0], str):
        print('ERROR: variable type of elements in unpriv must be str')
        return
    
    df = data.copy()

    priv_df = df
    priv_name = ''
    unpriv_df = df
    unpriv_name = ''
    
    if len(data) == 0:
        print('hello')
        
    for col1 in priv:
        priv_df = priv_df.loc[priv_df[col1] == 1]
        priv_name = priv_name+', '+col1
        
    if len(priv_df) == 0:
        print('ERROR: there is no data with given privileged columns')
        return
    
    for col2 in unpriv:
        unpriv_df = unpriv_df.loc[unpriv_df[col2] == 1]
        unpriv_name = unpriv_name+', '+col2
    
    if len(unpriv_df) == 0:
        print('ERROR: there is no data with given privileged columns')
        return
    
    priv_name = priv_name[2:]
    unpriv_name = unpriv_name[2:]
    
    priv_tn, priv_fp, priv_fn, priv_tp = confusion_matrix(priv_df[label], 
                                                          priv_df[pred]).ravel()
    priv_tpr = priv_tp/(priv_tp+priv_fn)
    priv_fpr = priv_fp/(priv_fp+priv_tn)
    priv_fnr = 1-priv_tpr

    unpriv_tn, unpriv_fp, unpriv_fn, unpriv_tp = confusion_matrix(unpriv_df[label], 
                                            unpriv_df[pred]).ravel()
    unpriv_tpr = unpriv_tp/(unpriv_tp+unpriv_fn)
    unpriv_fpr = unpriv_fp/(unpriv_fp+unpriv_tn)
    unpriv_fnr = 1-unpriv_tpr

    eod = unpriv_tpr - priv_tpr
    aod = ((unpriv_fpr - priv_fpr) + (unpriv_tpr - priv_tpr)) / 2

    priv_prob = len(priv_df.loc[(priv_df[pred] == 1)])/len(priv_df)
    unpriv_prob = len(unpriv_df.loc[(unpriv_df[pred] == 1)])/len(unpriv_df)
    di = unpriv_prob/priv_prob
    
    if verbosity >= 1:
        print('\n------------------------------Fairness tests results------------------------------\n')
        N = 2
        ind = np.arange(N) 
        width = 0.35
        plt.bar(ind, [priv_fpr, unpriv_fpr], width, label='False Positive Rate')
        plt.bar(ind + width, [priv_fnr, unpriv_fnr], width, label='False Negative Rate')
        plt.ylabel('Rate')
        plt.title('False Positive and False Negative Rate')
        plt.xticks(ind + width / 2, [priv_name, unpriv_name])
        plt.legend(loc='best')
        plt.show()
                
        
        print('\n------------------------------Equal Opportunity Difference------------------------------\n')
        if eod < -0.1: 
            print('Based on the equal opportunity difference, this model implies higher benefit for the privileged group')
        if eod > 0.1:
            print('Based on the equal opportunity difference, this model implies higher benefit for the unprivileged group')
        if verbosity >= 2:
            fig = plt.figure()
            ax = fig.add_axes([0,0,0.8,0.8])
            ax.set_ylim([-1, 1])
            ax.bar([''], [eod], width=0.5)
            ax.grid(color='#808080', linestyle='--', linewidth=1, axis='y', alpha=0.7)
            plt.title('Equal Opportunity Difference')
            if eod < 0:
                ax.text(-0.02, eod - 0.1, str(round(eod, 2)))
            else:
                ax.text(-0.02, eod + 0.1, str(round(eod, 2)))
            plt.show()
            print('Fairness for the equal opportunty difference metric is between -0.1 and 0.1 with the ideal value at 0')
        
        print('\n------------------------------Average Odds Difference------------------------------\n')
        if aod < -0.1:
            print('Based on the average odds difference, this model implies higher benefit for the privileged group')
        if aod > 0.1:
            print('Based on the average odds difference, this model implies higher benefit for the unprivileged group')
        if verbosity >= 2:
            fig = plt.figure()
            ax = fig.add_axes([0,0,0.8,0.8])
            ax.set_ylim([-1, 1])
            ax.bar([''], [aod], width=0.5)
            ax.grid(color='#808080', linestyle='--', linewidth=1, axis='y', alpha=0.7)
            plt.title('Average Odds Difference')
            if aod < 0:
                ax.text(-0.02, aod - 0.1, str(round(aod, 2)))
            else:
                ax.text(-0.02, aod + 0.1, str(round(aod, 2)))
            plt.show()
            print('Fairness for the average odds difference metric is between -0.1 and 0.1 with the ideal value at 0')
            if verbosity >= 3:
                df2 = pd.DataFrame({'Priviledged': [priv_tpr, priv_fpr], 'Unpriviledged': [unpriv_tpr, unpriv_fpr]}, index=['TPR', 'FPR'])
                print('')
                print(df2)
        
        print('\n------------------------------Disparate Impact------------------------------\n')
        if di < 0.8:
            print('Based on the disparate impact, this model implies higher benefit for the privileged group')
        if di > 1.2:
            print('Based on the disparate impact, this model implies higher benefit for the unprivileged group')
        if verbosity >= 2:
            fig = plt.figure()
            ax = fig.add_axes([0,0,0.8,0.8])
            if di < 2:
                ax.set_ylim([0, 2])
            else: 
                ax.set_ylim([0, round(di+0.5)])
            ax.bar([''], [di], width=0.5)
            ax.grid(color='#808080', linestyle='--', linewidth=1, axis='y', alpha=0.7)
            plt.title('Disparate Impact')
            ax.text(-0.02, di + 0.1, str(round(di, 2)))
            plt.show()
            print('Fairness for the disparate impact metric is between 0.8 and 1.2 with the ideal value at 1')
            if verbosity >= 3:
                df3 = pd.DataFrame({'Priviledged': [priv_prob], 'Unpriviledged': [unpriv_prob]}, index=['Probability of predicted value = 1'])
                print('')
                print(df3)
    return