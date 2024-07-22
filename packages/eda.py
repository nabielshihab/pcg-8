import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def check_portion(df, column, **kwargs):
    """
    Show the number and the portion of each value in a column.

    Parameters
    ---------
    df : {dataframe} a pandas dataframe
    column : {str} column name
    **kwargs : plotting keyword arguments
    """
    # plot number of samples in each status category 
    df[column].value_counts().plot(kind='bar', ylabel='Num. of Samples', **kwargs)

    # the percentage of each loan status
    # keep in mind that `display` method only applicable on a jupyter notebook.
    display(pd.DataFrame({'# Samples': df[column].value_counts(), 
                  'Percentage': df[column].value_counts() / df.shape[0] * 100}))
    

def show_null_containings(df):
    """
    Show the number and the portion of null values in each column in descending order.
    Only null-containing columns are shown. 
    
    Parameter
    ---------
    df : {dataframe} a pandas dataframe
    """
    df_null = pd.DataFrame(df.isna().sum(), columns=['Percentage of Nulls'])
    df_null['Percentage of Nulls'] = df_null / df.shape[0] * 100
    
    
    display(df_null[df_null['Percentage of Nulls'] > 0].sort_values('Percentage of Nulls', ascending=False))
    
    
def show_one_unique_containings(df):
    """
    Show the list of features which contain only one unique value.
    
    Parameter
    ---------
    df : {dataframe} a pandas dataframe
    """
    cols = df.nunique().sort_values()
    cols = cols[cols==1]

    display(pd.DataFrame(cols, columns=['Num of Unique Values']))


def plot_corr_matrix(corrMatrix, annot, **kwargs):
    """
    plot correlation matrix

    Parameters:
    -----------
    corrMatrix : {dataframe} a correlation matrix dataframe. 
    annot : {bool} if True, the correlation coefficient will be displayed.
 
    Returns:
    -----------
    fig : {matplotlib.Figure} A correlation matrix heatmap      
    ax : {matplotlib.AxesSubplot} an axes subplot
    """
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corrMatrix, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(nrows=1, ncols=1, **kwargs)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corrMatrix, mask=mask, cmap=cmap, vmax=1, center=0, vmin=-1,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = annot, ax=ax)

    return fig, ax
    
    
def high_cc_features(corrMatrix, threshold):
    """
    Get highly correlated features (absolute CC greater than or equal to the desired threshold) 
    
    Parameter:
    -----------
    corrMatrix : {dataframe} correlation matrix dataframe.
    threshold : {float} correlation coefficient threshold.
    
    Returns:
    -----------
    df : {dataframe} dataframe of pairs of highly correlated features
    
    """
    # unstack correlation matrix
    corrMatrix = corrMatrix.unstack()
    idx = np.abs(corrMatrix) >= threshold
    df = pd.DataFrame(corrMatrix.loc[:, idx])
    
    # remove rows which contain the same elem1 and elem2
    df.index.names = ['feature_1','feature_2']
    df.reset_index(inplace=True)
    df.rename(columns={0: 'CC'}, inplace=True)
    df = df[df['feature_1'] != df['feature_2']]

    # keep 1 row which contain the same elements in elem1 and elem2 columns (K-Si, Si-K)
    df['temp'] = df['feature_1']+df['feature_2'] 
    for idx in df.index:
        df.loc[idx, 'temp'] = ''.join(sorted(df.loc[idx, 'temp']))
    df.drop_duplicates(subset='temp', inplace=True)
    df.drop('temp', axis=1, inplace=True)

    # reset index
    df.reset_index(inplace=True, drop=True)
    
    return df


def plot_loan_portion(df, category, xlabel, **kwargs):
    """
    plot normalized portions of charged off and fully paid loans for a certain category.
    sorted by Charged Off loans in ascending order.
    
    Parameters:
    -----------
    df : {dataframe} a dataframe.
    category : {str} a categorical variable existed in the dataframe.
    xlabel : {str} x-axis label.
    **kwargs : plotting keyword arguments
    
    """
    df = pd.pivot_table(df, index=category, columns='loan_status', margins=True, aggfunc='count')
    df = pd.DataFrame(df.iloc[:-1, 0:2].values, columns=['Charged Off', 'Fully Paid'], index=df.index[:-1]).T

    # normalize the value
    df = df / df.sum()
    
    # transpose
    df = df.T

    # plot
    fig, ax = plt.subplots()
    df.sort_values('Charged Off').plot(kind='bar', stacked=True, ylabel='Normalized Portion of Loans', xlabel=xlabel, ax=ax, color=['red', 'blue'], alpha=.5, **kwargs)
    ax.legend(bbox_to_anchor=(1.3, 1), loc='upper right', borderaxespad=0)
