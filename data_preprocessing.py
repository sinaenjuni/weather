import pathlib
import pandas as pd

COLUMNS = ['time','STN','Lon','Lat','isitu-LST','Band1','Band2','Band3','Band4',
                'Band5','Band6','Band7','Band8','Band9','Band10','Band11','Band12','Band13',
                'Band14','Band15','Band16','30daysBand3','30daysBand13','GK2A-LST','SolarZA',
                'SateZA','ESR','Height','LandType','instiu-TA','instiu-HM','instiu-TD','instiu-TG',
                'instiu-TED0.05','instiu-TED0.1','instiu-TED0.2','instiu-TED0.3','instiu-TED0.5',
                'instiu-TED1.0','instiu-TED1.5','instiu-TED3.0','instiu-TED5.0','instiu-PA','instiu-PS']

BATH_PTAH = '/workspace/data/unzipfiles'
path = pathlib.Path(BATH_PTAH)

dfs = []
for i in list(path.glob("202001/*.csv")):
    print(i.name)
    data = pd.read_csv(i)
    data.columns = COLUMNS
    dfs.append(data)


data = pd.concat(dfs, ignore_index=True)
data.mean(0)
data.sum(0)
data.min(0)
data.max(0)
data.std(0)
print(dfs.keys())

for i in dfs:
    print(i.keys())