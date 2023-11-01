import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

df = pd.read_csv('https://raw.githubusercontent.com/daniloui001/Correlation/main/movies.csv')

#Overview

df.head()