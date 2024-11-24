# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:09:06 2024

@author: Matt
"""

import requests
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import numpy as np
import re




#-----FUNCTION to scrape kal valley website and return all URLs for tournament results pages----------
def func_ScrapeKalValley(base_url):
    #----get all URLS conraining 'results'
    # Function to check if a URL is allowed to be crawled based on robots.txt
    def is_url_allowed(url, base_url):
        robots_url = urljoin(base_url, '/robots.txt')
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch('*', url)
    
    # Function to extract all links from a page
    def get_all_links(url, base_url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                links = []
                for a_tag in soup.find_all('a', href=True):
                    link = urljoin(base_url, a_tag['href'])  # Ensure absolute URL
                    if base_url in link:  # Ensure the link is within the same domain
                        links.append(link)
                return set(links)  # Return unique links
            else:
                print(f"Failed to retrieve the page: {url}")
                return set()
        except Exception as e:
            print(f"Error retrieving links from {url}: {e}")
            return set()
    
    # Function to filter links based on keywords like "results"
    def filter_result_links(links, keyword="results"):
        return [link for link in links if keyword in link.lower()]
    
    # Main function to crawl the website and find result URLs
    def find_result_urls(base_url):
        all_links = get_all_links(base_url, base_url)
        result_links = filter_result_links(all_links, keyword="results")
        
        # Check if each link is allowed by robots.txt
        allowed_links = [link for link in result_links if is_url_allowed(link, base_url)]
        
        return allowed_links  # Return the list of allowed URLs
    
    # Function to return URLs in a DataFrame
    def urls_to_dataframe(urls):
        df = pd.DataFrame(urls, columns=["Result URLs"])
        return df

    
    # Find all result-related URLs while respecting robots.txt
    result_urls = find_result_urls(base_url)
    
    # Convert to DataFrame
    Result_Pages = urls_to_dataframe(result_urls)
    
    return Result_Pages
#------------------------------------------------------------------------------------------        
    
    
    
    
    
#-----------FUNCTION to scrape tournament results from result page URL----------
def func_FetchResults(url):
    try:
        # Send GET request to fetch the page content
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all tables on the page
            tables = soup.find_all('table')
            
            if tables:
                
                # Iterate over each table and convert it into a pandas DataFrame
                for i, table in enumerate(tables):
                    # Get all rows from the table
                    rows = table.find_all('tr')
                    
                    # Extract the headers, if present
                    headers = [th.get_text(strip=True) for th in rows[0].find_all('th')] if rows else None
                    
                    # Extract the table data row by row
                    table_data = []
                    for row in rows[1:]:  # Skip the header row if it exists
                        columns = row.find_all('td')
                        row_data = [col.get_text(strip=True) for col in columns]
                        table_data.append(row_data)
                    
                    # Convert to a DataFrame
                    if headers:
                        Results = pd.DataFrame(table_data, columns=headers)
                    else:
                        Results = pd.DataFrame(table_data)              
            else:
                print("No tables found on the page.")
        else:
            print(f"Failed to retrieve the page: {url}")
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        
    return Results
#--------------------------------------------------------------------------------------------------




#------------FUNCTION to find the coordinates of a string in a dataframe------
def func_FindIndex(df,text):
    coordinates = [(row_idx, col) for col in df.columns 
               for row_idx in df.index 
               if text.strip().lower() == re.sub(r'[^a-zA-Z\s]', '',str(df.at[row_idx, col])).strip().lower()] #remove any accidental special charachters from text, strip trialing/leading whitespace, convert to lowercase
    return coordinates
#----------------------------------------------------------------------------





#------FUNCTION to find the neighbor one column to the right of a string in a dataframe-----
def func_FindNeighbor(df,text):
    positions = func_FindIndex(df,text) 
    if len(positions)==0:
        neighbor = ''
    else:
        neighbor = df.at[positions[0][0],positions[0][1]+1]
    return neighbor
#----------------------------------------------------------------------------------





#---------FUNCTION to parse an annual results page into a stats and results dataframe-------------
def func_ParseResults(Results):
    
    #separate results page out into each tourney (lake)
    rowcounts = (Results != "").sum(axis=1)
    lakeIdx = []
    for idx in range(len(rowcounts)-1):
        count1 = rowcounts[idx]
        count2 = rowcounts[idx+1]
        if (count1 == 1) and (count2 > 1):
            lakeIdx.append(idx)       
    lakeIdx.append(len(rowcounts)+1)

    tourneyStats = pd.DataFrame(columns = ['LakeName','Date','BigBass','TotalWeight','WinningWeight',
                                           'AverageWeight','AverageBigBass','AverageFishPerBoat','Fish','Boats','Dead','Smallies','Smallie Percentage'])
    tourneyResults = pd.DataFrame(columns = ['Place','Fish','BigBass','Weight','LakeName','Date'])
    #---process each tourney (lake)
    for idx in range(len(lakeIdx)-1):   
        # 1) separate lake metadata from tourney results data
        AllData = Results.loc[lakeIdx[idx]:lakeIdx[idx+1]-2].reset_index(drop=True)
        PlaceCoors = func_FindIndex(AllData,'Place')[0]
        lakeData = AllData.loc[0:PlaceCoors[0]-1,:]
        tourneyData = AllData.loc[PlaceCoors[0]:]
        
        # 2) parse out tourney specific lake metadata
        lakeNameIdx = lakeData.columns[lakeData.iloc[0].astype(bool)][0] #find the column containing the lake name - can use for date also
        lake = lakeData.loc[0,lakeNameIdx].lower()
        try:
            datematches = lakeData.map(lambda x: x if isinstance(x, str) and re.search('.+,.+,.+', x) else np.nan).dropna(axis=1,how='all').dropna(axis=0,how='all')
            date = datetime.strptime(datematches.iloc[0,0], "%a, %b %d, %y")
        except Exception:
            datematches = lakeData.map(lambda x: x if isinstance(x, str) and re.search(r'\d{2}/\d{2}/\d{2}', x) else np.nan).dropna(axis=1,how='all').dropna(axis=0,how='all')
            date = datetime.strptime(datematches.iloc[0,0], "%m/%d/%y")
            
        BB = func_FindNeighbor(lakeData,'Big Bass')
        weight = func_FindNeighbor(lakeData,'Weight')
        fish = func_FindNeighbor(lakeData,'Fish')
        boats = func_FindNeighbor(lakeData,'Boats')
        dead = func_FindNeighbor(lakeData,'Dead')
        smallies = func_FindNeighbor(lakeData,'Smallies')
        
        # 3) parse full tourney results
        idx1 = func_FindIndex(tourneyData,'Place')[-1][1]
        idx2 = func_FindIndex(tourneyData,'Fish')[-1][1]
        idx3 = func_FindIndex(tourneyData,'Big Bass')[-1][1]
        idx4 = func_FindIndex(tourneyData,'Weight')[-1][1]
        rowIdx = func_FindIndex(tourneyData,'Place')[-1][0]
        tourneyRow = tourneyData.loc[rowIdx+1:,[idx1,idx2,idx3,idx4]]
        tourneyRow.replace('','0',inplace=True)
        tourneyRow.columns = ['Place','Fish','BigBass','Weight']
        tourneyRow['Place'] = tourneyRow['Place'].astype(int)
        tourneyRow['Fish'] = tourneyRow['Fish'].astype(int)
        tourneyRow['BigBass'] = tourneyRow['BigBass'].astype(float)
        tourneyRow['Weight'] = tourneyRow['Weight'].astype(float)
        tourneyRow['LakeName'] = lake
        tourneyRow['Date'] = date
        
        
        #3a) redundant checks for stats entries  
        fishTR = sum(tourneyRow['Fish'])
        if abs(fishTR-float(fish)) > 0:
            print(f"mismatch in FISH for {lake} on {date} \n {fish} (original) vs {fishTR} (calculated)")
            fish = fishTR #correct fish
        weightTR = sum(tourneyRow['Weight'])
        if abs(weightTR-float(weight)) > 3:
            print(f"mismatch in WEIGHT for {lake} on {date} \n {weight} (original) vs {weightTR} (calculated)")
            weight = weightTR #correct weight
        boatsTR = len(tourneyRow)
        if abs(boatsTR-float(boats)) > 0:
            print(f"mismatch in BOATS for {lake} on {date} \n {boats} (original) vs {boatsTR} (calculated)")
            boats = boatsTR #correct boats
        bigbassTR = max(tourneyRow['BigBass'])
        if abs(bigbassTR-float(BB)) > 0.1:
            print(f"mismatch in BIG BASS for {lake} on {date} \n {BB} (original) vs {bigbassTR} (calculated)")
            BB = bigbassTR #correct big bass
        
        
        
        
        
        # 4) append values to dataframes
        winningweight = tourneyRow['Weight'].iloc[0]
        avgweight = np.mean(tourneyRow['Weight'])
        avgBB = np.mean(tourneyRow.loc[tourneyRow['BigBass']>0,'BigBass'])
        try:
            avgFPB = float(fish)/float(boats)
        except Exception:
            avgFPB = None
        try:
            smalliePct = 100*float(smallies)/float(fish)
        except Exception:
            smalliePct = 0
                
        myRow = [lake,date,BB,weight,winningweight,avgweight,avgBB,avgFPB,fish,boats,dead,smallies,smalliePct]
        tourneyStats.loc[len(tourneyStats),:] = myRow
        tourneyRow = tourneyRow.dropna(axis=1, how='all')  
        tourneyResults = tourneyResults.dropna(axis=1, how='all')  
        tourneyResults = pd.concat([tourneyRow,tourneyResults],ignore_index=True)
        

        
    return tourneyStats, tourneyResults
#-----------------------------------------------------------------------------------------






#----------MAIN---------------------      
base_url = 'https://www.kal-valley.com/'

#get all URLs for results pages
Results_urls = func_ScrapeKalValley(base_url)


FinalStats = pd.DataFrame()
FinalResults = pd.DataFrame()

#scrape each tournament results page and aggregate the data
for url in Results_urls['Result URLs']:
    tempresults = func_FetchResults(url)
    Tstats,Tresults = func_ParseResults(tempresults)
    FinalStats = pd.concat([FinalStats,Tstats],ignore_index=True)
    FinalResults = pd.concat([FinalResults,Tresults],ignore_index=True)
    
    
#lake name standardizations
keepNames = ['fine lake','long lake (portage)','materson lake','croton pond']
changeNames = ['fine lake – mystery','long lake','mystery lake – materson','croton pond – 2022 saturday classic']
mapping = dict(zip(changeNames, keepNames))
FinalStats['LakeName'] = FinalStats['LakeName'].replace(changeNames,keepNames)
FinalResults['LakeName'] = FinalResults['LakeName'].replace(mapping)  

#convert '0' smallie data prior to 2017 when it wasn't tracked to nan
FinalStats.loc[FinalStats['Date'] < datetime(2018,1,1),'Smallies'] = None
FinalStats.loc[FinalStats['Date'] < datetime(2018,1,1),'Smallie Percentage'] = None


    
# save
FinalStats.to_pickle('FinalStats2024.pkl')
FinalResults.to_pickle('FinalResults2024.pkl')
    
    
    
    
    
    
    
    
