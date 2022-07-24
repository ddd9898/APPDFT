import numpy as np
import pandas as pd

#Code 1:Load MHC pseudo labels
pseudo_dir = './data/NetMHCpan_train/MHC_pseudo.dat' 
temp = pd.read_csv(pseudo_dir,header=None).values.tolist()
pseudoMHC_Dic_NetMHCpan = dict()
for item in temp:
    MHCname,pseudoMHC = item[0].split()
    if 'HLA-' in MHCname:
        MHCname = MHCname.replace('HLA-A','HLA-A*')
        MHCname = MHCname.replace('HLA-B','HLA-B*')
        MHCname = MHCname.replace('HLA-C','HLA-C*')
    pseudoMHC_Dic_NetMHCpan[MHCname] = pseudoMHC

#
replace_MHC = {
'HLA-A*3':'HLA-A*3001',
'HLA-A*1':'HLA-A*0101',
'HLA-A*11':'HLA-A*1101',
'HLA-B*44:01':'HLA-B*44:02',
'HLA-B*08:011':'HLA-B*08:11',
}
#Check if every MHC in training data can be found in pseudoMHC_Dic_NetMHCpan
dataset = list()
full_data = pd.read_csv("./data/ELIM_full_trainingData.csv")
BA_data = pd.read_csv("./data/BA_full_trainingData.csv")
HLA_list = [item for item in full_data['MHC']]
HLA_list.extend([item for item in BA_data['MHC']])
HLA_list = np.unique(HLA_list).tolist()

#Add HLA from pmhc test dataset
HLA_list.append('HLA-C*14:03')

#Add HLA from Mhcflurry test dataset which never showed in training data of NetMHCpan4.1
HLA_list.append('HLA-A*34:02')
HLA_list.append('HLA-A*36:01')
HLA_list.append('HLA-B*07:04')
HLA_list.append('HLA-B*35:07')
HLA_list.append('HLA-C*03:02')
HLA_list.append('HLA-C*04:03')

not_find = False
for HLA in HLA_list:
    if HLA not in pseudoMHC_Dic_NetMHCpan.keys():
        print("{} miss!".format(HLA),end=' ')
        if HLA  in replace_MHC.keys():
            newHLA = replace_MHC[HLA]
            print("Replace {} with {}".format(HLA,newHLA))
            dataset.append([HLA,pseudoMHC_Dic_NetMHCpan[newHLA]])
        else:
            print("")
            not_find = True
    else:
        dataset.append([HLA,pseudoMHC_Dic_NetMHCpan[HLA]])
if not not_find:
    print("{} MHCs all find their pseudo sequence.".format(len(HLA_list)))
    
        
#Save to local
column=['MHC','pseudoSeq']
output_dir = './data/pseudoSequence(ELIM).csv'
output = pd.DataFrame(columns=column,data=dataset)
output.to_csv(output_dir,index = None)



# #Code 2:Load MHC pseudo labels
# pseudo_dir = './data/NetMHCpan_train/MHC_pseudo.dat' 
# temp = pd.read_csv(pseudo_dir,header=None).values.tolist()
# pseudoMHC_Dic_NetMHCpan = dict()
# for item in temp:
#     MHCname,pseudoMHC = item[0].split()
#     if 'HLA-' in MHCname:
#         MHCname = MHCname.replace('HLA-A','HLA-A*')
#         MHCname = MHCname.replace('HLA-B','HLA-B*')
#         MHCname = MHCname.replace('HLA-C','HLA-C*')
#     pseudoMHC_Dic_NetMHCpan[MHCname] = pseudoMHC

# #
# replace_MHC = {
# "HLA-A*01": "HLA-A*01:01",
# "HLA-A*02": "HLA-A*02:01",
# "HLA-A*02:01:110": "HLA-A*02:01",
# "HLA-A*02:01:48": "HLA-A*02:01",
# "HLA-A*02:01:59": "HLA-A*02:01",
# "HLA-A*02:01:98": "HLA-A*02:01",
# "HLA-A*02:15": "HLA-A*0215",
# "HLA-A*03" :"HLA-A*03:01",
# "HLA-A*11" :"HLA-A*11:01",
# "HLA-A*24:02:84":"HLA-A*24:02",
# "HLA-B*07":"HLA-B*07:02",
# "HLA-B*08":"HLA-B*08:01",
# "HLA-B*08:01:29":"HLA-B*08:01",
# "HLA-B*15":"HLA-B*15:01",
# "HLA-B*18":"HLA-B*18:01",
# "HLA-B*27":"HLA-B*27:01",
# "HLA-B*27:05:31":"HLA-B*27:05",
# "HLA-B*35":"HLA-B*35:01",
# "HLA-B*35:08:01":"HLA-B*35:08",
# "HLA-B*35:42:01":"HLA-B*35:42",
# "HLA-B*42":"HLA-B*42:01",
# "HLA-B*44:03:08":"HLA-B*44:03",
# "HLA-B*44:05:01":"HLA-B*44:05",
# "HLA-B*57":"HLA-B*57:01",
# "HLA-B*58":"HLA-B*58:01",
# "HLA-E*01:01:01:03":"HLA-E01:01",
# 'HLA-A*24':'HLA-A*24:02',
# 'HLA-B*81':'HLA-B*81:01'
# }
# #Check if every MHC in training data can be found in pseudoMHC_Dic_NetMHCpan
# dataset = list()
# full_data = pd.read_csv("./data/trainingData.csv")
# HLA_list = np.unique(full_data['MHC']).tolist()

# #Add HLA from pmhcTCR test dataset
# HLA_list.append('HLA-A*24')
# HLA_list.append('HLA-B*81')

# not_find = False
# for HLA in HLA_list:
#     if HLA not in pseudoMHC_Dic_NetMHCpan.keys():
#         print("{} miss!".format(HLA),end=' ')
#         if HLA  in replace_MHC.keys():
#             newHLA = replace_MHC[HLA]
#             print("Replace {} with {}".format(HLA,newHLA))
#             dataset.append([HLA,pseudoMHC_Dic_NetMHCpan[newHLA]])
#         else:
#             print("")
#             not_find = True
#     else:
#         dataset.append([HLA,pseudoMHC_Dic_NetMHCpan[HLA]])
# if not not_find:
#     print("{} MHCs all find their pseudo sequence.".format(len(HLA_list)))
    
        
# #Save to local
# column=['MHC','pseudoSeq']
# output_dir = './data/pseudoSequence.csv'
# output = pd.DataFrame(columns=column,data=dataset)
# output.to_csv(output_dir,index = None)


