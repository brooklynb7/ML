import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output

datasetFolder = "datasets/nba-players-stats-since-1950"
print(check_output(["ls", datasetFolder]).decode("utf8"))

players=pd.read_csv(datasetFolder + '/Players.csv')
Seasons=pd.read_csv(datasetFolder + '/Seasons_Stats.csv')

print(players.head())