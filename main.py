import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import graphviz
import numpy as np


def get_df():
    # Create new collumn for champions to be added to data frame
    Champions = ["Los Angeles Lakers (2002)", "San Antonio Spurs (2003)", "Los Angeles Lakers (2004)",
                 "San Antonio Spurs (2005)", "Dallas Mavericks (2006)", "San Antonio Spurs (2007)",
                 "Los Angeles Lakers (2008)", "Los Angeles Lakers (2009)", "Los Angeles Lakers (2010)",
                 "Dallas Mavericks (2011)", "Oklahoma City Thunder (2012)", "San Antonio Spurs (2013)",
                 "San Antonio Spurs (2014)", "Golden State Warriors (2015)", "Golden State Warriors (2016)",
                 "Golden State Warriors (2017)", "Golden State Warriors (2018)", "Golden State Warriors (2019)",
                 "Los Angeles Lakers (2020)"]

    # Get initial data fram just from 2002 season
    df = pd.read_html('https://www.espn.com/nba/stats/team/_/season/2002/seasontype/2')
    df = pd.concat([df[0], df[1]], axis=1, ignore_index=True)
    for i in range(len(df)):
        df.loc[i:i, (1,)] = df.loc[i:i, (1,)] + " (2002)"

    # Get data from rest of seasons (up to 2021)
    for i in range(2003, 2021):
        tmp_df = pd.read_html('https://www.espn.com/nba/stats/team/_/season/' + str(i) + '/seasontype/2')
        tmp_df = pd.concat([tmp_df[0], tmp_df[1]], axis=1, ignore_index=True)
        for j in range(len(tmp_df)):
            tmp_df.loc[j:j, (1,)] = tmp_df.loc[j:j, (1,)] + " (" + str(i) + ")"
        df = pd.concat([df, tmp_df], axis=0, ignore_index=True)

    # Add Champions Collumn
    df['Champion'] = np.zeros((len(df),), dtype=int)
    for i in range(len(df)):
        if df[1][i] in Champions:
            df.loc[i:i, ('Champion',)] = 1

    return df

def get_test_df():
    my_dict = {}
    df = pd.read_html('https://www.espn.com/nba/stats/team/_/season/2021/seasontype/2')
    df = pd.concat([df[0], df[1]], axis=1, ignore_index=True)
    num_data = df.loc[:, range(3, 21)]
    num_data = num_data.values.tolist()
    for i in range(len(df)):
        my_dict[df[1][i]] = num_data[i]
    return my_dict


if __name__ == '__main__':
    # Get data
    data = get_df()
    curr = get_test_df()
    #print(data)
    print(curr)


    # Separate feature and target collumns
    X = data[range(3, 21)]
    y = data['Champion']

    #print(X)
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    data = tree.export_graphviz(dtree, out_file=None, feature_names=range(3, 21))
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')

    img = pltimg.imread('mydecisiontree.png')
    imgplot = plt.imshow(img)
    plt.show()

    print("Champion is among the following: ")

    for key in curr:
        if 1 in dtree.predict([curr[key]]):
            print(key)


"""
old code

import pandas as pd


def get_table_data():
    #Get three seperate tables from player data on 3 separate pages
    nba_2021_data = \
        pd.read_html('https://basketball.realgm.com/nba/stats/2021/Averages/Qualified/points/All/desc/1/Regular_Season')
    nba_table1 = nba_table1_data[12]

    nba_table2_data = \
        pd.read_html('https://basketball.realgm.com/nba/stats/2021/Averages/Qualified/points/All/desc/2/Regular_Season')
    nba_table2 = nba_table2_data[12]

    nba_table3_data = \
        pd.read_html('https://basketball.realgm.com/nba/stats/2021/Averages/Qualified/points/All/desc/3/Regular_Season')
    nba_table3 = nba_table3_data[12]

    # Put tables together vertically, clean up, and return finalized table
    finalized_table = pd.concat([nba_table1, nba_table2, nba_table3], axis=0, ignore_index=True)
    return finalized_table


if __name__ == '__main__':
    table = get_table_data()
    stat_3_mean = round(table["3P%"].mean(), 4)
    stat_3_std = round(table["3P%"].std(), 4)

    print("Mean: " + str(stat_3_mean))
    print("STD DEV: " + str(stat_3_std))

    #wow all within 1 stddev
    tracker = 0
    for player in table["3P%"]:
        if player < stat_3_mean + (1 * stat_3_std) or player > stat_3_mean - (1 * stat_3_std):
            tracker += 1
    print(tracker)
    print(len(table))


"""




