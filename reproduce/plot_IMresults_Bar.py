import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

##Sars-Cov2-con test compare
x_names = ('Recall', 'Precision','F1 Score')
y_values = [[1.000000,0.114286,0.205128], #APPDFT
            [0.14,0.05,0.07368], #DeepHLApan
            [0.375000,0.02,0.03797],  #IEDB
            [0.875000,0.114754,0.202899], #DeepImmuno-CNN
            [1.000000,0.087912,0.161616], #NetMHCpan4.1
            [1.000000,0.087912,0.161616], #TransPHLA
            ]

bar_width = 0.1
x_values = list()
x_values.append(np.arange(len(x_names)))
for n in range(len(y_values)):
    x_values.append(x_values[-1]+bar_width)

# Draw
plt.figure(4)
nouseIMdata = ['','','','','*','*']
# label = ['APPDFT','DeepImmuno-CNN','DeepHLApan','IEDB','NetMHCpan4.1','TransPHLA']
label = ['APPDFT','DeepHLApan','IEDB','DeepImmuno-CNN','NetMHCpan4.1','TransPHLA']
# colors = ['r','lime','g','dodgerblue','orange','yellow']
colors = [(239/255, 65/255, 67/255),(145/255, 213/255, 66/255),
          (31/255, 146/255, 139/255),(75/255, 101/255, 175/255),
          (244/255, 111/255, 68/255),(248/255, 230/255, 32/255)]
palettes = {'APPDFT':'#fb8072','NetMHCpan4.1':'#80b1d3','NetMHCpan4.0':'#bebada',
            'TransPHLA':'#fccde5','DeepHLApan':'#fdb462','IEDB':'#ffffb3',
            'DeepHLApan +\n NetMHCpan4.0':'#ccebc5','DeepImmuno-CNN':'#b3de69'}

for n in range(len(y_values)):
    plt.bar(x_values[n], height=y_values[n], width=bar_width, color=palettes[label[n]], label=label[n])
    for m in range(3):
        plt.text(x_values[n][m],y_values[n][m],nouseIMdata[n],ha='center')
plt.legend() 
plt.xticks(x_values[2] + bar_width/2, x_names) 
plt.ylabel('')
plt.title('SARS-Cov2 unexposed')

plt.pause(1)
plt.savefig('./output/figures/Recall and Precision curve benchmark on Sars-Cov2-un testing data.png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/Recall and Precision curve benchmark on Sars-Cov2-un testing data.pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片




##Sars-Cov2-con test compare
x_names = ('Recall', 'Precision','F1 Score')
y_values = [[0.880000,0.314286,0.463158], #APPDFT
            [0.400000,0.285714,0.333333], #DeepHLApan
            [0.52,0.25,0.33766], #IEDB
            [0.680000,0.278689,0.395349], #DeepImmuno-CNN
            [1.000000,0.274725,0.431034], #NetMHCpan4.1
            [1.000000,0.274725,0.431034], #TransPHLA
            ]

bar_width = 0.1
x_values = list()
x_values.append(np.arange(len(x_names)))
for n in range(len(y_values)):
    x_values.append(x_values[-1]+bar_width)

# Draw
plt.figure(3)
nouseIMdata = ['','','','','*','*']
# label = ['APPDFT','DeepImmuno-CNN','DeepHLApan','IEDB','NetMHCpan4.1','TransPHLA']
label = ['APPDFT','DeepHLApan','IEDB','DeepImmuno-CNN','NetMHCpan4.1','TransPHLA']
# colors = ['r','g','dodgerblue','lime','yellow','orange']
colors = [(239/255, 65/255, 67/255),(31/255, 146/255, 139/255),
          (75/255, 101/255, 175/255),(145/255, 213/255, 66/255),
          (248/255, 230/255, 32/255),(244/255, 111/255, 68/255)]

for n in range(len(y_values)):
    plt.bar(x_values[n], height=y_values[n], width=bar_width, color=palettes[label[n]], label=label[n])
    for m in range(3):
        plt.text(x_values[n][m],y_values[n][m],nouseIMdata[n],ha='center')

plt.legend() 
plt.xticks(x_values[2] + bar_width/2, x_names) 
plt.ylabel('')
plt.title('SARS-Cov2 convalescent')

plt.pause(1)
plt.savefig('./output/figures/Recall and Precision curve benchmark on Sars-Cov2-con testing data.png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/Recall and Precision curve benchmark on Sars-Cov2-con testing data.pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片





##TESLA test compare
x_names = ('Recall', 'Precision','F1 Score')
y_values = [[0.8,0.090323,0.162319], #APPDFT
            [0.342857,0.057692,0.098765], #DeepHLApan
            [0.628571,0.076389,0.136223], #IEDB
            [0.828571,0.085294,0.154667], #DeepImmuno-CNN
            [1.000000,0.076419,0.141988], #NetMHCpan4.1
            [1.000000,0.071429,0.133333], #TransPHLA
            ] 

bar_width = 0.1
x_values = list()
x_values.append(np.arange(len(x_names)))
for n in range(len(y_values)):
    x_values.append(x_values[-1]+bar_width)

# Draw
plt.figure(2)
# label = ['APPDFT','DeepImmuno-CNN','DeepHLApan','IEDB','NetMHCpan4.1','TransPHLA']
label = ['APPDFT','DeepHLApan','IEDB','DeepImmuno-CNN','NetMHCpan4.1','TransPHLA']
# colors = ['r','lime','g','yellow','dodgerblue','orange']
colors = [(239/255, 65/255, 67/255),(145/255, 213/255, 66/255),
          (31/255, 146/255, 139/255),(248/255, 230/255, 32/255),
          (75/255, 101/255, 175/255),(244/255, 111/255, 68/255)]

nouseIMdata = ['','','','','*','*']
for n in range(len(y_values)):
    plt.bar(x_values[n], height=y_values[n], width=bar_width, color=palettes[label[n]], label=label[n])
    for m in range(3):
        plt.text(x_values[n][m],y_values[n][m],nouseIMdata[n],ha='center')

plt.legend() 
plt.xticks(x_values[2] + bar_width/2, x_names) 
plt.ylabel('')
plt.title('TESLA')

plt.pause(1)
plt.savefig('./output/figures/Recall and Precision curve benchmark on TESLA testing data.png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/Recall and Precision curve benchmark on TESLA testing data.pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片






##IM test compare
x_names = ('Recall', 'Precision','F1 Score')
y_values = [
            [0.875000, 0.543689,0.670659], #APPDFT
            [0.593750,0.015664,0.030523], #DeepHLApan
            [0.922,0.366,0.523994], #NetMHCpan4.0
            [0.953125,0.271111,0.422145], #NetMHCpan4.1
            [0.968750,0.114391,0.204620], #TransPHLA
            [0.547,0.486,0.5147], #DeepHLApan + NetMHCpan4.0
            ]

bar_width = 0.1
x_values = list()
x_values.append(np.arange(len(x_names)))
for n in range(len(y_values)):
    x_values.append(x_values[-1]+bar_width)

# Draw
plt.figure(1)
nouseIMdata = ['','','','*','*','*','']
# label = ['APPDFT','DeepHLApan +\n NetMHCpan4.0','DeepHLApan','NetMHCpan4.0','NetMHCpan4.1','TransPHLA']
label = ['APPDFT','DeepHLApan','NetMHCpan4.0','NetMHCpan4.1','TransPHLA','DeepHLApan +\n NetMHCpan4.0']
# colors = ['dodgerblue','g','r','lime','yellow','orange']
# colors = [(75/255, 101/255, 175/255),(31/255, 146/255, 139/255),
#           (239/255, 65/255, 67/255),'#1383C2',
#           (240/255, 240/255, 150/255),(244/255, 111/255, 68/255)]


for n in range(len(y_values)):
    plt.bar(x_values[n], height=y_values[n], width=bar_width, color=palettes[label[n]], label=label[n])
    for m in range(3):
        plt.text(x_values[n][m],y_values[n][m],nouseIMdata[n],ha='center')

plt.legend() 
plt.xticks(x_values[2] + bar_width/2, x_names) 
plt.ylabel('') 
plt.title('Independent IM test data')

plt.pause(1)
plt.savefig('./output/figures/Recall and Precision curve benchmark on IM testing data.png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/Recall and Precision curve benchmark on IM testing data.pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片
