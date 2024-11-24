# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:53:54 2024

@author: Matt
"""

import streamlit as st
import pandas as pd
import datetime as dtime
import plotly.graph_objects as go
import numpy as np



#-------------------------------FUNCTIONS-----------------------------------------------------------------------------------------------------
#-------function to load and cache data-------
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
#change sidebar header font
st.markdown(
    """
    <style>
    .css-1d391kg {
        font-size: 24px;  
        text-decoration: underline; 
    }
    </style>
    """,
    unsafe_allow_html=True)

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write(f"Tournaments Held: {len(TS)}")
st.sidebar.write(f"Lakes Fished: {len(TS['LakeName'].unique())}")
st.sidebar.write(f"Fish Caught: {sum(TR['Fish'])} ({round(sum(TR['Weight']))} lbs!)")
st.sidebar.write(f"Biggest Bass to Date: {max(TR['BigBass'])} lbs")



#---LAKE STATISTICS TAB
with tab1:
    #---data slicer input widgets
    col1,col2,col3 = st.columns([0.6,0.2,0.2])
    with col1:
        lakes = st.multiselect(label='Lake Selection',options=['All']+sorted(TS['LakeName'].unique()),default='All',max_selections=6)
    with col2:
        months = st.multiselect(label='Month(s)',options=['All','May','June','July','August','September','October'],default='All')
    with col3:
        dateRange = st.date_input(label='Date Range',value=[dtime.datetime(2010,1,1),dtime.datetime(2025,1,1)])

    #---create "metric to visualize" widget (column 1)
    col1a, col2a = st.columns([0.15,0.85],vertical_alignment='center')
    with col1a:
        c1 = st.container() #container for plot options
        with c1:
            #add radio buttons for 1st plot metric selection
            metric = st.radio(label='Metric to Visualize',options=['Winning Bag Weight','Big Bass Weight','Average Bag Weight',
                                                          'Average Big Bass Weight','Average Fish per Boat','Total Fish Weight','Total Fish',
                                                          'Number of Boats','Total Fish Dead','Total Smallmouth','Percent Smallmouth'],index=0)
            
            #increase font size for radio button title
            st.markdown(
            """<style>
            div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
                font-size: 22px;
                text-decoration: underline;
            }
            </style>
            """, unsafe_allow_html=True)      
    
            #map radio button options to dataframe columns
            columnMappings = {'Winning Bag Weight': ['WinningWeight','Lbs'],
                              'Big Bass Weight': ['BigBass','Lbs'],
                              'Average Bag Weight': ['AverageWeight','Lbs'],
                              'Average Big Bass Weight': ['AverageBigBass','Lbs'],
                              'Total Fish Weight': ['TotalWeight','Lbs'],
                              'Total Fish': ['Fish','Fish'],
                              'Number of Boats': ['Boats','Boats'],
                              'Average Fish per Boat':['AverageFishPerBoat','Fish'],
                              'Total Fish Dead': ['Dead','Fish'],
                              'Total Smallmouth': ['Smallies','Fish'],
                              'Percent Smallmouth':['Smallie Percentage','%']}
                    
            metric1 = columnMappings[metric][0]
            units = columnMappings[metric][1]
            
            
        #---add "show averages" toggle
        showAvgLines = st.toggle(label='Show Averages',value=False)
   
        
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
    
    
    #---create scatter plot (column 2)
    with col2a:
        fig_line_plot = go.Figure() #create figure 
        bgc = 'black'
        gc = 'dimgrey'
        colors = ['steelblue','indianred','mediumseagreen','goldenrod','rebeccapurple','darkslategrey']
        if (len(lakes) < 7) and ('All' not in lakes): # if there are 6 or less lakes selected, plot data in different colors as line plots, otherwise, mass scatter    
            legStatus = True    
            for idx,lake in enumerate(lakes):
                dtemp = TS3[TS3['LakeName']==lake].sort_values(by='Date')
                X = dtemp['Date']
                Y = dtemp[metric1]
                fig_line_plot.add_trace(go.Scatter(mode='lines+markers',x=X, y=Y, name=lake, line=dict(color=colors[idx],width=3), marker=dict(color=colors[idx],size=8)))
                if showAvgLines:
                    fig_line_plot.add_hline(y=np.mean(Y), line=dict(color=colors[idx], width=2, dash="dash"))
                
        else:   
            X = TS3['Date']
            Y = TS3[metric1]
            fig_line_plot.add_trace(go.Scatter(mode='markers',x=X, y=Y,  marker=dict(color=colors[0],size=6)))
            if showAvgLines:
                fig_line_plot.add_hline(y=np.mean(Y), line=dict(color=colors[0], width=2, dash="dash"))
            legStatus = False
              
        fig_line_plot.update_layout(xaxis=dict(type='date',color=gc),showlegend=legStatus)
        fig_line_plot.update_xaxes(gridcolor=gc,mirror=True,ticks='outside',showline=True,title=dict(text='Date',font=dict(color='white')),tickfont=dict(color='white'))
        fig_line_plot.update_yaxes(gridcolor=gc,mirror=True,ticks='outside',showline=True,title=dict(text=units,font=dict(color='white')),tickfont=dict(color='white'))
        fig_line_plot.update_layout(title_text=f"<u>{metric} vs. Date</u>",plot_bgcolor=bgc,margin=dict(l=60, r=60, t=80, b=40))
        fig_line_plot.update_layout(title=dict(yanchor='top',y=0.9,font=dict(size=24)))
        
        st.plotly_chart(fig_line_plot,use_container_width=True,theme=None) #main line plot
    

    #---create spider web plot (col1b)      
    col1b, col2b = st.columns([0.50,0.50],vertical_alignment='center')
    with col1b:
        if len(lakes)>0:
            fig_radar_plot = go.Figure() #create figure 
            #set colors
            bgc = 'black'
            gc = 'dimgrey'
            colors = ['steelblue','indianred','mediumseagreen','goldenrod','rebeccapurple','darkslategrey']
            #set metrics for radar chart and corresponding dataframe column names
            radarMetrics = ['Winning Bag Weight','Average Bag Weight','Big Bass Weight','Average Big Bass Weight','Average Fish per Boat','Winning Bag Weight']
            radarCols = [columnMappings[x][0] for x in radarMetrics]
            #get max/min values for each column to scale radar chart data to effectively set the range
            metricRangesHigh = [1.1*max(TS3.loc[TS3[col].notna(),col]) for col in radarCols]
            metricRangesLow = [0.9*min(TS3.loc[TS3[col].notna(),col]) for col in radarCols]
            if (len(lakes) < 7) and ('All' not in lakes): # if there are 6 or less lakes selected, plot data in different colors as line plots, otherwise, mass scatter    
                legStatus = True  
                for idx,lake in enumerate(lakes): #create a trace for each lake             
                    dtemp = TS3[TS3['LakeName']==lake].sort_values(by='Date')
                    #average data for each metric, ignoring nonetypes
                    Y = []
                    for col in radarCols:
                        Y.append(round(np.mean(dtemp.loc[dtemp[col].notna(), col]),2))
                    Y.append(Y[0])
                    Yscaled = [(v-rl)/(rh-rl) for v, rh, rl in zip(Y, metricRangesHigh, metricRangesLow)]            
                    fig_radar_plot.add_trace(go.Scatterpolar(mode='lines+markers',r=Yscaled,theta=radarMetrics, name=lake, 
                                                             line=dict(color=colors[idx],width=2.5), marker=dict(color=colors[idx],size=7),
                                                             fillcolor=bgc,text=Y,hovertemplate='%{text}'))
            else:
                legStatus = False
                Y = []
                for col in radarCols:
                    Y.append(round(np.mean(TS3.loc[TS3[col].notna(), col]),2))
                Y.append(Y[0])
                Yscaled = [(v-rl)/(rh-rl) for v, rh, rl in zip(Y, metricRangesHigh, metricRangesLow)]            
                fig_radar_plot.add_trace(go.Scatterpolar(mode='lines+markers',r=Yscaled,theta=radarMetrics,name='All', 
                                                         line=dict(color=colors[0],width=2.5), marker=dict(color=colors[0],size=7),
                                                         fillcolor=bgc,text=Y,hovertemplate='%{text}'))
                    
            fig_radar_plot.update_layout(plot_bgcolor=bgc,margin=dict(l=80, r=80, t=80, b=40),
                                         polar=dict(
                                            radialaxis=dict(ticks='',color=gc),
                                            angularaxis=dict(color=gc,tickfont=dict(color='white'))))      
            fig_radar_plot.update_polars(radialaxis_showticklabels=False,bgcolor=bgc)
            fig_radar_plot.update_layout(title_text='<u>Average Metrics by Lake</u>', title=dict(yanchor='top',y=0.95,font=dict(size=22)),showlegend=legStatus)
            
            st.plotly_chart(fig_radar_plot,use_container_width=True,theme=None) #radar plot
            
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        










