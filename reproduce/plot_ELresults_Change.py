import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

np.random.seed(100)

# Code 1: Load NetMHCpan4.1 result
results_path_a = './data/NetMHCpan_test/supplementary_table_8.xlsx'
results_a =pd.read_excel(results_path_a,keep_default_na=False).values.tolist()

box_data = list()
# Method_a = 'MHCFlurry_EL'
Method_a = 'NetMHCpan-4.1'
idx = results_a[0].index(Method_a)

MHC_list = list()
AUCROC_Method_a = list()
for n in range(2,len(results_a)):
    MHC = results_a[n][0]
    AUC_ROC = results_a[n][idx]
    
    MHC_list.append(MHC)
    AUCROC_Method_a.append(AUC_ROC)

# # #Load our results a
# AUCROC_Method_a = np.zeros(36)
# results_path = './output/results/evaluations_ELdata_Model-nolink_index1(5folds).csv'
# # results_path = './output/results/evaluations_ELdata_baseline-EL_index0(5folds).csv'
# results =pd.read_csv(results_path,keep_default_na=False).values.tolist()
# MHC_list = list()
# for item in results:
#     MHC = item[0].replace('*','')
#     MHC_list.append(MHC)
# for item in results:
#     MHC = item[0].replace('*','')
#     AUC_ROC = item[5]
    
#     MHC_idx = MHC_list.index(MHC)
#     AUCROC_Method_a[MHC_idx] = AUC_ROC

#Load our results b
AUCROC_Method_b = np.zeros(len(MHC_list))
results_path = './output/results/evaluations_ELdata_Model-link_index0(5folds).csv'
results =pd.read_csv(results_path,keep_default_na=False).values.tolist()
for item in results:
    MHC = item[0].replace('*','')
    AUC_ROC = item[1]
    
    MHC_idx = MHC_list.index(MHC)
    AUCROC_Method_b[MHC_idx] = AUC_ROC
    
    
diff = np.array(AUCROC_Method_b) - np.array(AUCROC_Method_a)

count_better = 0
for item in diff:
    if item > 0:
        count_better += 1
print("{}/36".format(count_better))

lens = 20
x_shift = -lens*np.ones(36)*np.sign(diff)
y_shift = lens*np.ones(36)*np.sign(diff)
for n in range(36):
    if x_shift[n]>0:
        x_shift[n] = 2 * x_shift[n]

# ##Label cor for baseline-EL and Model-link
# y_shift[14] = -10 #B0702

# ##Label cor for netMHCpan4.1 and Model-link
x_shift[32] = 50 #C0501
y_shift[32] = -70 #C0501
y_shift[14] = -30 #B0702
y_shift[13] = -20 #A6801
x_shift[5] = -110 #A2301
y_shift[5] = 20 #A2301
x_shift[7] = -120 #A2601
y_shift[7] = 30 #A2601
x_shift[22] = -200 #B3503
y_shift[22] = 40 #B3503
x_shift[30] = -70 #B5801
y_shift[30] = 55 #B5801
x_shift[15] = 40 #B0801
y_shift[15] = 24 #B0801
x_shift[9] = 60 #A3002
y_shift[9] = -70 #A3002
x_shift[33] = 50 #C0702
y_shift[33] = -50 #C0702
x_shift[13] = -70 #A6801
y_shift[13] = -70 #A6801
x_shift[31] = 110 #C0303
y_shift[31] = 20 #C0303

plt.figure(figsize=(10, 15), dpi=100)
ax = plt.gca()
ax.set_aspect(1)

plt.plot([0, 1],[0, 1], c='k', linestyle='--')
plt.plot([0, 1],[0.01, 1.01], c='g', linestyle='--')
plt.plot([0, 1],[-0.01, 0.99], c='r', linestyle='--')
plt.legend(['Equal','+0.01','- 0.01'],prop = {'size':12})

plt.scatter(AUCROC_Method_a, AUCROC_Method_b, c='m', s=10)
# new_texts = [plt.text(x_, y_, text, fontsize=6) for x_, y_, text in zip(AUCROC_Method_a, AUCROC_Method_b, MHC_list)]
# adjust_text(new_texts,arrowprops=dict(arrowstyle='->',color='blue',lw=0.5))


for label, x, y,x_s,y_s,d in zip(MHC_list, AUCROC_Method_a, AUCROC_Method_b,x_shift,y_shift,diff):
    if np.abs(d)<0.01:
        continue
    if d > 0:
        plt.annotate(
            label,
            xy=(x, y), xytext=(x_s,y_s),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc=(31/255, 146/255, 139/255), alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
            fontsize = 15)
    else:
        plt.annotate(
            label,
            xy=(x, y), xytext=(x_s,y_s),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc=(239/255, 65/255, 67/255), alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
            fontsize = 15)







plt.xlim(0.75, 1)
plt.ylim(0.75, 1)
plt.xlabel("NetMHCpan4.1", fontdict={'size': 16})
# plt.xlabel("baseline-ELIM", fontdict={'size': 16})
# plt.xlabel("baseline-EL", fontdict={'size': 16})
plt.ylabel("APPDFT", fontdict={'size': 16})
plt.title("AUC ROC", fontdict={'size': 20})

# plt.get_current_fig_manager().window.state('zoomed') #Max window

plt.savefig('./output/figures/AUC compare on EL testing data.png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/AUC compare on EL testing data.pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片

# plt.savefig('./output/figures/AUC compare on EL testing data(self compare1).png',bbox_inches = 'tight')
# plt.savefig('./output/figures/AUC compare on EL testing data(self compare1).pdf',bbox_inches = 'tight',format='pdf')  # 保存图片



#Code 2:
# #Load our results a and b
AUCROC_Method_a = np.zeros(36)
AUCROC_Method_b = np.zeros(36)
results_path_a = './output/results/evaluations_ELdata_baseline-ELIM_index0(5folds).csv'
results_path_b = './output/results/evaluations_ELdata_baseline-EL_index0(5folds).csv'
results_a =pd.read_csv(results_path_a,keep_default_na=False).values.tolist()
results_b =pd.read_csv(results_path_b,keep_default_na=False).values.tolist()
MHC_list = list()
for item in results_a:
    MHC = item[0].replace('*','')
    MHC_list.append(MHC)
for item in results_a:
    MHC = item[0].replace('*','')
    AUC_ROC = item[1]
    
    MHC_idx = MHC_list.index(MHC)
    AUCROC_Method_a[MHC_idx] = AUC_ROC
    
for item in results_b:
    MHC = item[0].replace('*','')
    AUC_ROC = item[1]
    
    MHC_idx = MHC_list.index(MHC)
    AUCROC_Method_b[MHC_idx] = AUC_ROC

#Load our results c
AUCROC_Method_c = np.zeros(len(MHC_list))
results_path = './output/results/evaluations_ELdata_Model-link_index0(5folds).csv'
results =pd.read_csv(results_path,keep_default_na=False).values.tolist()
for item in results:
    MHC = item[0].replace('*','')
    AUC_ROC = item[1]
    
    MHC_idx = MHC_list.index(MHC)
    AUCROC_Method_c[MHC_idx] = AUC_ROC
    
    
diff_c_b = np.array(AUCROC_Method_c) - np.array(AUCROC_Method_b)
diff_c_a = np.array(AUCROC_Method_c) - np.array(AUCROC_Method_a)

plt.figure(dpi=300,figsize=(24,8))
plt.bar(np.arange(len(diff_c_a))*2+0.6, height=diff_c_a, width=0.6,color = '#80AFBF')
plt.bar(np.arange(len(diff_c_b))*2, height=diff_c_b, width=0.6,color = '#EFDBB9')

plt.xlim([-5,75])
plt.xlabel('HLA')
plt.ylabel('AUC change')
plt.legend(['APPDFT compare to baseline-ELIM','APPDFT compare to baseline-EL'])
plt.plot([-5, 75],[0.01, 0.01], c='g', linestyle='--')
plt.plot([-5, 75],[-0.01, -0.01], c='r', linestyle='--')


for mhc, diff_cb, diff_ca in zip(MHC_list, diff_c_b, diff_c_a):
    
    
    if np.abs(diff_cb) > np.abs(diff_ca):
        y = diff_cb
    else:
        y = diff_ca
    x = MHC_list.index(mhc) * 2
    
    plt.text(x-1,y,mhc)
a = 1

plt.savefig('./output/figures/AUC compare on EL testing data(self compare3).png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/AUC compare on EL testing data(self compare3).pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片
