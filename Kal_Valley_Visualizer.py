# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:55:04 2024

@author: Matt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


TS = pd.read_pickle('S:\\Python Projects\\Bass Tournament Web Scraping\\FinalStats2023.pkl')
TS.replace('','0',inplace=True)
TS.iloc[:, 2:] = TS.iloc[:, 2:].astype(float)
TR = pd.read_pickle('S:\\Python Projects\\Bass Tournament Web Scraping\\FinalResults2023.pkl')



Lakes = sorted(TS['LakeName'].unique())

for lake in Lakes:
    dtemp = TS[TS['LakeName']==lake].sort_values(by='Date')
    
    # Create a figure with 2 vertical subplots
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
    fig.suptitle(lake, fontsize=16)
    
    
    # Plotting on the first subplot
    axs[0].plot(dtemp['Date'],dtemp['BigBass'],color='rebeccapurple',marker='.')
    axs[0].axhline(np.mean(dtemp['BigBass']),color='rebeccapurple',linestyle='--')
    axs[0].set_title('Big Bass')
    axs[0].grid()  # Optional: adds a grid for better visualization
    
    # Plotting on the second subplot
    axs[1].plot(dtemp['Date'],dtemp['WinningWeight'],color='green',marker='.')
    axs[1].axhline(np.mean(dtemp['WinningWeight']),color='green',linestyle='--')
    axs[1].set_title('WinningWeight')
    axs[1].grid()  # Optional: adds a grid for better visualization
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plots
    plt.show()