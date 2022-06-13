import os
import pathlib

import numpy as np
import pandas as pd

COLUMNS = ['time','STN','Lon','Lat','isitu-LST','Band1','Band2','Band3','Band4',
                'Band5','Band6','Band7','Band8','Band9','Band10','Band11','Band12','Band13',
                'Band14','Band15','Band16','30daysBand3','30daysBand13','GK2A-LST','SolarZA',
                'SateZA','ESR','Height','LandType','instiu-TA','instiu-HM','instiu-TD','instiu-TG',
                'instiu-TED0.05','instiu-TED0.1','instiu-TED0.2','instiu-TED0.3','instiu-TED0.5',
                'instiu-TED1.0','instiu-TED1.5','instiu-TED3.0','instiu-TED5.0','instiu-PA','instiu-PS']
TARGETS = ['GK2A-LST', 'instiu-TA']


BATH_PTAH = '/workspace/data/unzipfiles'
bast_path = pathlib.Path(BATH_PTAH)

STN_PATH = '/workspace/data/stn_data/target1/'
stn_path = pathlib.Path(STN_PATH)

SAVE_PATH = './savefiles/ori_month/'

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)

all_col_data = []
lst_col_data = []
ta_col_data = []
for m in list(bast_path.glob("*")):
    dfs = []
    for f in list(m.glob("*.csv")):
        print(m.name, f)
        df = pd.read_csv(f)
        df.columns = COLUMNS
        dfs.append(df)
    dfs = pd.concat(dfs, ignore_index=True)

    all_col = dfs[~(dfs.isin([-999]).mul(1).sum(1) > 0)]
    lst_col = dfs[~dfs[TARGETS[0]].isin([-999])]
    ta_col  = dfs[~dfs[TARGETS[1]].isin([-999])]

    all_col_data.append(all_col)
    lst_col_data.append(lst_col)
    ta_col_data.append(ta_col)
    print(len(dfs), len(all_col_data), len(lst_col_data), len(ta_col_data))


all_col_data = pd.concat(all_col_data, ignore_index=True)
lst_col_data = pd.concat(lst_col_data, ignore_index=True)
ta_col_data = pd.concat(ta_col_data, ignore_index=True)
all_col_data.to_csv(SAVE_PATH + "all_col_data.csv")
lst_col_data.to_csv(SAVE_PATH + "lst_col_data.csv")
ta_col_data.to_csv(SAVE_PATH + "ta_col_data.csv")
# data.to_csv(SAVE_PATH + f"{m.name}.csv")


lst_col_data.__len__()
ta_col_data.__len__()

np.intersect1d(lst_col_data["STN"].unique(), ta_col_data["STN"].unique()).__len__()
np.union1d(lst_col_data["STN"].unique(), ta_col_data["STN"].unique()).__len__()


list(stn_path.glob("*"))


dfs = []
for i in list(path.glob("202001/*.csv")):
    print(i.name)
    # data = pd.read_csv(i)
    # data.columns = COLUMNS
    # dfs.append(data)

data = pd.concat(dfs, ignore_index=True)
all_col = data[~(data.isin([-999]).mul(1).sum(1) > 0)]

all_col[TARGETS[0]]
all_col[TARGETS[1]]
all_col.mean(0)
all_col.sum(0)
all_col.max(0)
all_col.std(0)
all_col.min(0)

corr = all_col.corr()
corr = corr[[TARGETS[0], TARGETS[1]]]
import seaborn as sns
import matplotlib.pyplot as plt
colormap = plt.cm.PuBu
sns.set(font_scale=2)

plt.figure(figsize=(70, 70))

sns.heatmap(corr,
            linewidths = 0.1,
            vmax = 1.0,
           square = True,
            cmap = colormap,
            linecolor = "white",
            annot = True,
            annot_kws = {"size" : 16}
           )

plt.show()



len(data)

lst_col = data[~data[TARGETS[0]].isin([-999])]
lst_col[~(lst_col.isin([-999]).mul(1).sum(1) > 0)]


lst_col.mean(0)
lst_col.sum(0)
lst_col.max(0)
lst_col.std(0)
lst_col.min(0)


data[TARGETS[1]].isin([-999]).value_counts()


ta_col = data[~data[TARGETS[1]].isin([-999])]
ta_col.min(0)

ta_col[~(ta_col.isin([-999]).mul(1).sum(1) > 0)]


t = data[~data[TARGETS[0]].isin([-999])]
t.min(0)

t = data[~data[TARGETS[1]].isin([-999])]
t.min(0)


data.mean(0)
data.sum(0)
data.max(0)
data.std(0)
print(dfs.keys())

for i in dfs:
    print(i.keys())


