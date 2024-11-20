# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:53:54 2024

@author: Matt
"""

import streamlit as st
import pandas as pd
import datetime as dtime
import plotly.graph_objects as go




#-------------------------------FUNCTIONS-----------------------------------------------------------------------------------------------------
@st.cache_data
def func_LoadData():
    #---load and prepare data
    TS = pd.read_pickle('S:\\Python Projects\\Bass Tournament Web Scraping\\FinalStats2024.pkl')
    TS.replace('','0',inplace=True)
    TS[['BigBass','TotalWeight','WinningWeight','AverageWeight','AverageBigBass','Fish','Boats','Dead','Smallies']] = TS[['BigBass','TotalWeight','WinningWeight','AverageWeight','AverageBigBass','Fish','Boats','Dead','Smallies']].astype(float)
    TR = pd.read_pickle('S:\\Python Projects\\Bass Tournament Web Scraping\\FinalResults2024.pkl')
    #sandardize datetime channels
    TS['Date'] = pd.to_datetime(TS['Date'])
    TR['Date'] = pd.to_datetime(TR['Date'])
    
    return TR,TS
    



#--------------------------MAIN SCRIPT---------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------
#initialize dashboard
st.set_page_config(page_title='Bass Tournament Results Visualizer', layout="wide")

#load data
TR,TS = func_LoadData()

#initialize tabs
tab1,tab2 = st.tabs(['Lake Statistics','Tournament Results Explorer'])
with tab1:
    #st.title("Lake Statistics")
    st.subheader('Explore historic tournament and lake statistics',divider='grey')

with tab2:
    #st.title("Tournament Results Explorer")
    st.subheader('Drill down into individual tournament results',divider='grey')
    
#create sidebar + sidebar stats
st.sidebar.header("Bass Tournament Result Dashboard Overview")
st.sidebar.write("Welcome to the bass tournament result dashboard. This tool provides fishing insights from years of bass"
                     " tournament data in the southwest Michigan area.  \n  \n The 'Laket Statistics' tab will display general tourmanent specific stats"
                     " (such as big bass, winning weight, number of boats, ect.).  \n  \n The 'Tournament Results Explorer' tab will allow the user to dig into the full results"
                         " from each tournament.  \n  \n  Please know any personal information that could identify individuals has been removed from the dataset."
                         "  \n  \n Happy exploring!")

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write(f"Tournaments Held: {len(TS)}")
st.sidebar.write(f"Lakes Fished: {len(TS['LakeName'].unique())}")
st.sidebar.write(f"Fish Caught: {sum(TR['Fish'])} ({round(sum(TR['Weight']))} lbs!)")
st.sidebar.write(f"Biggest Bass to Date: {max(TR['BigBass'])} lbs")



#---Create plot options
with tab1:
    #data slicers
    col1,col2,col3 = st.columns([0.6,0.2,0.2])
    with col1:
        lakes = st.multiselect(label='Lake Selection',options=['All']+sorted(TS['LakeName'].unique()),default='All')
    with col2:
        months = st.multiselect(label='Month(s)',options=['All','May','June','July','August','September','October'],default='All')
    with col3:
        dateRange = st.date_input(label='Date Range',value=[dtime.datetime(2010,1,1),dtime.datetime(2025,1,1)])

    #y variable select and plot
    col1a, col2a = st.columns([0.15,0.85])
    with col1a:
        metric = st.radio(label='Metric to Visualize',options=['Winning Weight','Big Bass','Average Weight',
                                                      'Average Big Bass','Total Weight','Total Fish',
                                                      'Number of Boats','Total Dead','Total Smallies'],index=0)
        metric1 = "".join(metric.split())
    
    #--prep data for plotting
    #apply lake slicer
    if 'All' not in lakes: 
        TS1 = TS[TS['LakeName'].isin(lakes)]
    else:
        TS1 = TS
        
    #apply date range slicer    
    TS2 = TS1[(TS1['Date']>=pd.to_datetime(dateRange[0])) & (TS1['Date']<=pd.to_datetime(dateRange[1]))] 
    
    #apply month slicer
    if 'All' not in months:
        DFmonths = TS2['Date'].dt.month_name()
        TS3 = TS2[DFmonths.isin(months)]
    else:
        TS3 = TS2
    
    #create scatter plot
    with col2a:
        fig_line_plot = go.Figure() #create figure 
        bgc = 'black'
        gc = 'silver'
        if len(lakes) < 9: # if there are 8 or less lakes selected, plot data in different colors as line plots, otherwise, mass scatter
            colors = ['steelblue','indianred','mediumseagreen','goldenrod','rebeccapurple','orange','slategray','aquamarine']
            for idx,lake in enumerate(lakes):
                dtemp = TS3[TS3['LakeName']==lake].sort_values(by='Date')
                X = dtemp['Date']
                Y = dtemp[metric1]
                fig_line_plot.add_trace(go.Scatter(mode='lines+markers',x=X, y=Y, name=lake, line=dict(color=colors[idx]), marker=dict(color=colors[idx])))
            
        else:   
            X = TS3['Date']
            Y = TS3[metric1]
            fig_line_plot.add_trace(go.Scatter(mode='markers',x=X, y=Y, name='Multiple',  marker=dict(color=colors[0],size=10)))
            
            
            
        fig_line_plot.update_layout(xaxis=dict(type='date',color=gc),showlegend=True)
        fig_line_plot.update_xaxes(gridcolor=gc,mirror=True,ticks='outside',showline=True)
        fig_line_plot.update_yaxes(gridcolor=gc,mirror=True,ticks='outside',showline=True)
        fig_line_plot.update_layout(title_text=f"{metric} vs. Date",plot_bgcolor=bgc,margin=dict(l=60, r=60, t=120, b=40))
        fig_line_plot.update_layout(title=dict(yanchor='top',y=0.8,font=dict(size=28)))
        st.plotly_chart(fig_line_plot,use_container_width=True,theme=None) #main line plot
    

            











