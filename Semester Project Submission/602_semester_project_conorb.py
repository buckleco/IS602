import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import numpy as np
import datetime

def obtain_soccer_data():
    #read in the log file from the working directory
    df = pd.read_csv ('EnglishFootballResults.csv', sep=',', header=0, quotechar='"',escapechar='=',dtype={'Date': object, 'division': object})
    return (df)

def obtain_weather_data():
    #read in the log file from the working directory
    df = pd.read_csv ('LondonWeatherStation.csv', sep=',', header=0, quotechar='"',escapechar='=')
    return (df)

def clean_soccer_data(df_soc):
    #I am only concerned with cleaning the date field (after removing the rows where it = NaN) and the home games of the London clubs
    #the total goals field ('totgoal') is fine and does not need to be touched here
    #remove all the rows before the 1993 season (there are no dates associated with games prior to 1993)
    df_soc = df_soc[(df_soc['Season']>1992)]
    #create a list of the London clubs in my data set
    ldnlst = ['Arsenal','Chelsea','Tottenham Hotspur','West Ham United','Fulham','Queens Park Rangers','Charlton Athletic','Millwall','Leyton Orient','Crystal Palace','Brentford']
    #remove all rows where the home team is not in the London list
    df_soc = df_soc[df_soc['home'].isin(ldnlst)]
    #fix the date format (this will be used later to merge the soccer data with the weather data)
    df_soc['Date'] = pd.to_datetime(df_soc['Date'], format='%m/%d/%Y')
    #return the cleaned up data frame
    return df_soc

def clean_weather_data(df_wea):
    #I am only concerned with two fields here - the Date & Precipitation fields
    #fix the date format (this will be used later to merge the soccer data with the weather data)
    df_wea[' YEARMODA'] = pd.to_datetime(df_wea[' YEARMODA'], format='%Y%m%d')
    #remove all of the precipitation with missing data (missing data is represented as 99.99)
    df_wea = df_wea[df_wea['PRCP  ']!="99.99 "]
    #extract the first 4 characters from the values in the PRCP field and put them into a new fields called 'PRECIP'
    df_wea['PRECIP'] = df_wea['PRCP  '].str[1:5]
    #convert the precipitation value (which no longer contains the alphabetic character) to a float
    df_wea['PRECIP'] = df_wea['PRECIP'].astype(float)
    #df_wea[['PRECIP']] = df_wea[['PRECIP']].astype(float)
    #return the cleaned up data frame
    return df_wea

def merge_data(df_soc_clean,df_wea_clean):
    #merge the two data sets on the date field performing an inner join (i.e. dropping the soccer rows with no weather data)
    df_final = pd.merge(df_soc_clean, df_wea_clean, left_on='Date', right_on=' YEARMODA', how='inner')
    #reduce the data frame to just 4 columns - Date, home team, total goals in the game and the quantity of precipitation
    df_final = df_final.reindex(columns=['Date', 'home', 'totgoal', 'PRECIP'])
    #sort the data by the date (increasing)
    df_final = df_final.sort(columns=['Date'])
    #return the final data frame
    return df_final

def clean_and_merge_data(df_soc,df_wea):

    #clean the soccer data
    df_soc_clean = clean_soccer_data(df_soc)
    #clean the weather data
    df_wea_clean = clean_weather_data(df_wea)
    #merge the soccer and weather data
    df_final = merge_data(df_soc_clean,df_wea_clean)
    #return the clean refined dataset
    return df_final

def aggregate_data(df_final,colp,colg,team=None):

    #remove the rows where the given team is not the home team
    if team is not None:
        df_final = df_final[df_final['home']==team]
    #df_final = df_final[df_final['home'].isin(team)]
    #reduce the data frame down to the total goals and precipitation columns
    df_final = df_final.reindex(columns=[colg, colp])
    #group the data by precipitation
    df_final = df_final.groupby(colp).mean()
    #reset the index after grouping the data, which now contains the average goals per precipitation level
    df_final = df_final.reset_index()
    #return the final data frame
    return df_final

def create_regression_results(df_overall,indvar,depvar):

    #this function takes in a data frame (containing x & y variables) and two text variables (the names of the x & y variables)
    result = sm.ols(formula=depvar + " ~ " + indvar, data=df_overall).fit()
    #it returns the standard statsmodels OLS output
    return result.summary()

def print_regession_results(regression_results, title):

    #this function prints out the OLS results as given along with a title
    print ""
    print "###########################################"
    print title
    print "###########################################"
    print ""
    print regression_results

def extract_xy_variables(df,x,y):

    #this function creates independent and dependent variables based on the input supplied
    ind = df[x]
    dep = df[y]
    return ind,dep

def plot_regressions(ind0,dep0,ind1,dep1,ind2,dep2,ind3,dep3,ind4,dep4):

    #this function creates the plots on all the independent and dependent variables provided
    #set the x & y axis label text variables
    xtext = 'daily precipitation (inches)'
    ytext = 'average goals per game'

    #set the plot window title
    fig = plt.figure('Soccer goals vs Precipitation regression plots : London based professional teams 1993 - 2010', figsize=(16, 7))
    #fig.suptitle('This is the subtitle', fontsize=20, y=1)

    #create the first subplot (overall)
    sub0 = plt.subplot(231)
    sub0.set_title('All Games',y=1.05)
    #fit with np.polyfit which returns a vector of coefficients that minimises the squared error.
    fit0 = np.polyfit(ind0,dep0,1)
    #extract the slope & intercept
    slope0 =  fit0[0]
    intercept0 = fit0[1]
    #fit_fn0 is a function which takes in the independent variable (ind) and returns an estimate for the dependent variable (dep)
    fit_fn0 = np.poly1d(fit0)
    plt.plot(ind0,dep0, 'go', ind0, fit_fn0(ind0), '--k')
    #set axis scales and labels
    plt.axis([0,1.2,0,5])
    plt.ylabel(ytext)
    plt.xlabel(xtext)
    #add the regression equation to the plot
    reg_lbl0 = "y = " + str("%.3f" % round(slope0,3)) + "x +" + str("%.2f" % round(intercept0,2))
    plt.text(0.55, 4, reg_lbl0)

    #create the second subplot (Arsenal)
    sub1 =plt.subplot(232)
    sub1.set_title('Arsenal Games',y=1.05)
    #fit with np.polyfit which returns a vector of coefficients that minimises the squared error.
    fit1 = np.polyfit(ind1,dep1,1)
    #extract the slope & intercept
    slope1 =  fit1[0]
    intercept1 = fit1[1]
    #fit_fn1 is a function which takes in the independent variable (ind) and returns an estimate for the dependent variable (dep)
    fit_fn1 = np.poly1d(fit1)
    plt.plot(ind1,dep1, 'ro', ind1, fit_fn1(ind1), '--k')
    #set axis scales and labels
    plt.axis([0,1.2,0,6])
    plt.ylabel(ytext)
    plt.xlabel(xtext)
    #add the regression equation to the plot
    reg_lbl1 = "y = " + str("%.3f" % round(slope1,3)) + "x +" + str("%.2f" % round(intercept1,2))
    plt.text(0.55, 4, reg_lbl1)

    #create the third subplot (Chelsea)
    sub2 = plt.subplot(233)
    sub2.set_title('Chelsea Games',y=1.05)
    #fit with np.polyfit which returns a vector of coefficients that minimises the squared error.
    fit2 = np.polyfit(ind2,dep2,1)
    #extract the slope & intercept
    slope2 =  fit2[0]
    intercept2 = fit2[1]
    #fit_fn2 is a function which takes in the independent variable (ind) and returns an estimate for the dependent variable (dep)
    fit_fn2 = np.poly1d(fit2)
    plt.plot(ind2,dep2, 'bo', ind2, fit_fn2(ind2), '--k')
    #set axis scales and labels
    plt.axis([0,1.2,0,9])
    plt.ylabel(ytext)
    plt.xlabel(xtext)
    #add the regression equation to the plot
    reg_lbl2 = "y = " + str("%.3f" % round(slope2,3)) + "x +" + str("%.2f" % round(intercept2,2))
    plt.text(0.55, 8, reg_lbl2)

    #create the fourth subplot (Tottenham Hotspur)
    sub3 = plt.subplot(234)
    sub3.set_title('Tottenham Games',y=1.05)
    #fit with np.polyfit which returns a vector of coefficients that minimises the squared error.
    fit3 = np.polyfit(ind3,dep3,1)
    #extract the slope & intercept
    slope3 =  fit3[0]
    intercept3 = fit3[1]
    #fit_fn3 is a function which takes in the independent variable (ind) and returns an estimate for the dependent variable (dep)
    fit_fn3 = np.poly1d(fit3)
    plt.plot(ind3,dep3, 'wo', ind3, fit_fn3(ind3), '--k')
    #set axis scales and labels
    plt.axis([0,1.2,0,7])
    plt.ylabel(ytext)
    plt.xlabel(xtext)
    #add the regression equation to the plot
    reg_lbl3 = "y = " + str("%.3f" % round(slope3,3)) + "x +" + str("%.2f" % round(intercept3,2))
    plt.text(0.55, 5, reg_lbl3)

    #create the fifth subplot (West Ham)
    sub4 = plt.subplot(235)
    #sub4.set_title('West Ham Games',horizontalalignment='center', verticalalignment='top')
    sub4.set_title('West Ham Games',y=1.05)
    #fit with np.polyfit which returns a vector of coefficients that minimises the squared error.
    fit4 = np.polyfit(ind4,dep4,1)
    #extract the slope & intercept
    slope4 =  fit4[0]
    intercept4 = fit4[1]
    #fit_fn4 is a function which takes in the independent variable (ind) and returns an estimate for the dependent variable (dep)
    fit_fn4 = np.poly1d(fit4)
    plt.plot(ind4,dep4, 'mo', ind4, fit_fn4(ind4), '--k')
    #set axis scales and labels
    plt.axis([0,1.2,0,7])
    plt.ylabel(ytext)
    plt.xlabel(xtext)
    #add the regression equation to the plot
    reg_lbl4 = "y = " + str("%.3f" % round(slope4,3)) + "x +" + str("%.2f" % round(intercept4,2))
    plt.text(0.5, 5, reg_lbl4)

    #create space between subplots
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    return plt

def main():
    #turn off warnings - specifically for SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    #obtain the soccer data
    df_soc = obtain_soccer_data()
    #obtain the weather data
    df_wea = obtain_weather_data()
    #clean the data and merge the two datasets into a dataframe containing just the fields needed for my analysis
    df_final = clean_and_merge_data(df_soc,df_wea)

    #store the column names of the x and y variables
    x_var = 'PRECIP'
    y_var = 'totgoal'

    #create the regression data frames (the total average goals [y] per given level of precipitation [x])
    df_overall = aggregate_data(df_final,x_var,y_var)
    df_arsenal = aggregate_data(df_final,x_var,y_var,'Arsenal')
    df_chelsea = aggregate_data(df_final,x_var,y_var,'Chelsea')
    df_tottenham = aggregate_data(df_final,x_var,y_var,'Tottenham Hotspur')
    df_westham = aggregate_data(df_final,x_var,y_var,'West Ham United')

    #run the regressions
    regression_results_overall = create_regression_results(df_overall,x_var,y_var)
    regression_results_arsenal = create_regression_results(df_arsenal,x_var,y_var)
    regression_results_chelsea = create_regression_results(df_chelsea,x_var,y_var)
    regression_results_tottenham = create_regression_results(df_tottenham,x_var,y_var)
    regression_results_westham = create_regression_results(df_westham,x_var,y_var)

    #print regression results and summary statistics
    print_regession_results(regression_results_overall, "ALL GAMES")
    print_regession_results(regression_results_arsenal, "ARSENAL GAMES")
    print_regession_results(regression_results_chelsea, "CHELSEA GAMES")
    print_regession_results(regression_results_tottenham, "TOTTENHAM GAMES")
    print_regession_results(regression_results_westham, "WEST HAM GAMES")

    #plot regression results
    #extract independent and dependent variables from each data frame
    ind0,dep0 = extract_xy_variables(df_overall,x_var,y_var)
    ind1,dep1 = extract_xy_variables(df_arsenal,x_var,y_var)
    ind2,dep2 = extract_xy_variables(df_chelsea,x_var,y_var)
    ind3,dep3 = extract_xy_variables(df_tottenham,x_var,y_var)
    ind4,dep4 = extract_xy_variables(df_westham,x_var,y_var)

    #create the regression plots
    plot_regressions(ind0,dep0,ind1,dep1,ind2,dep2,ind3,dep3,ind4,dep4)
    #show the plots
    plt.show()

# This is the main of the program.
if __name__ == "__main__":

    main()