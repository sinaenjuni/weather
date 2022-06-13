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


STNs = [ 90,  92,  93,  95,  96,  98,  99, 100, 101, 102, 104, 105, 106,
       108, 110, 112, 113, 114, 115, 116, 119, 121, 127, 128, 129, 130,
       131, 133, 135, 136, 137, 138, 139, 140, 142, 143, 146, 151, 152,
       153, 155, 156, 158, 159, 160, 161, 162, 163, 165, 167, 168, 169,
       170, 172, 174, 175, 177, 182, 184, 185, 188, 189, 192, 201, 202,
       203, 211, 212, 216, 217, 221, 226, 229, 230, 232, 235, 236, 238,
       239, 243, 244, 245, 247, 248, 251, 252, 253, 254, 255, 257, 258,
       259, 260, 261, 262, 263, 264, 266, 268, 271, 272, 273, 276, 277,
       278, 279, 281, 283, 284, 285, 288, 289, 294, 295, 296, 300, 301,
       302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 314, 315,
       316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328,
       329, 330, 349, 350, 351, 352, 353, 355, 356, 358, 359, 360, 361,
       364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
       377, 379, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410,
       411, 412, 413, 414, 415, 416, 417, 418, 419, 421, 423, 424, 425,
       426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438,
       439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451,
       452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464,
       465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
       478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490,
       491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503,
       504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516,
       517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 529, 530,
       531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543,
       544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556,
       557, 558, 559, 560, 561, 563, 565, 566, 567, 568, 569, 570, 571,
       572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 585,
       586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598,
       599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611,
       612, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625,
       626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638,
       639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651,
       652, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665,
       666, 667, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679,
       680, 681, 682, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696,
       697, 698, 699, 700, 701, 702, 703, 704, 706, 707, 708, 709, 710,
       711, 712, 713, 714, 716, 717, 718, 719, 720, 721, 722, 723, 724,
       725, 726, 727, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739,
       741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753,
       754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766,
       767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779,
       780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792,
       793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805,
       806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818,
       819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831,
       832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844,
       845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857,
       858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870,
       871, 872, 873, 874, 875, 876, 877, 878, 881, 882, 883, 884, 885,
       886, 888, 889, 890, 892, 893, 894, 895, 896, 897, 898, 899, 900,
       901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913,
       914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926,
       927, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940,
       941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 953, 954,
       955, 956, 957, 958, 959, 960, 961, 963, 964, 965, 966, 967, 970,
       972, 973, 974, 977, 978, 980, 984, 989, 990, 991, 992]

BATH_PTAH = '/workspace/data/unzipfiles'
bast_path = pathlib.Path(BATH_PTAH)

STN_PATH = '/workspace/data/stn_data/target1/'
stn_path = pathlib.Path(STN_PATH)

SAVE_PATH = './savefiles/ori_month/'

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)



data = []
for m in list(bast_path.glob("*")):
    dfs = []
    for f in list(m.glob("*.csv")):
        print(m.name, f)
        df = pd.read_csv(f)
        df.columns = COLUMNS
        dfs.append(df)
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv(SAVE_PATH + f"{m.name}.cvs")



all_col_data = []
lst_col_data = []
ta_col_data = []
for m in list(bast_path.glob("*"))[:1]:
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


