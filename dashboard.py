import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import re

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Load local CSV file
df = pd.read_csv('dataset/google-play-store-apps/googleplaystore.csv', encoding='ISO-8859-1').iloc[:2000]

# Check the column names of your dataframe
print(df.columns)

# List of columns that are numeric
numeric_indicators = ['Installs', 'Reviews', 'Rating']

# List of columns that can be used as x-axis
xaxis_indicators = df.columns

# Only keep rows where 'Installs' is not 'Free'
df = df.loc[df['Installs'] != 'Free']

# Only keep rows where 'Price' contains '$' or '0'
df = df.loc[df['Price'].str.contains('\$|0')]

# Remove '$' from 'Price' and convert to float
df['Price'] = df['Price'].apply(lambda x: re.sub("[^\d.]", "", x)).astype(float)

# Preprocess 'Size' column
df['Size'] = df['Size'].replace('Varies with device', '0')
df['Size'] = df['Size'].apply(lambda x: float(x.replace('M', '')) if 'M' in x else (float(x.replace('k', ''))/1000 if 'k' in x else None))

# Processing Installs column
df['Installs'] = df['Installs'].apply(lambda x: float(x.replace('+', '').replace(',', '')) if '+' in x or ',' in x else float(x))

df['Last Updated'] = pd.to_datetime(df['Last Updated'], format='%d-%b-%y')

# Convert 'Reviews' column to numeric
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

# Convert 'Rating' column to numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Remove rows with unrealistic 'Installs' values
df = df.loc[df['Installs'] < np.percentile(df['Installs'], 99)]

# Remove rows with unrealistic 'Reviews' values
df = df.loc[df['Reviews'] < np.percentile(df['Reviews'], 99)]

# Remove rows with unrealistic 'Rating' values
df = df.loc[(df['Rating'] >= 0) & (df['Rating'] <= 5)]

# List of columns that can be used as x-axis
xaxis_indicators = [col for col in df.columns if col not in ['App', 'Current Ver']] # Exclude 'App'

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in xaxis_indicators],
                value='Category'  # Set a default value
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in numeric_indicators],
                value='Rating'  # Set a default value
            ),
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-graph',
        ),
        html.Div([
            dcc.Graph(id='x-histogram'),
            dcc.Graph(id='y-histogram'),
        ], style={'display': 'inline-block', 'width': '45%', 'float': 'right', 'padding': '0 20'}),
    ], style={'width': '100%', 'display': 'flex', 'flex-direction': 'row', 'padding': '0 20'}),
], style={'display': 'flex', 'flex-direction': 'column'})

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-graph', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name, xaxis_type):
    if df[xaxis_column_name].dtype in ['int64', 'float64']:
        fig = {
            'data': [go.Scatter(
                x=df[xaxis_column_name],
                y=df[yaxis_column_name],
                text=df['App'],
                mode='markers',
                marker={
                    'size': 15,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            )],
            'layout': go.Layout(
                xaxis={'title': xaxis_column_name, 'type': 'log' if xaxis_type == 'Log' else 'linear'},
                yaxis={'title': yaxis_column_name},
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=450,
                hovermode='closest'
            )
        }
    else:
        if xaxis_column_name == 'Last Updated':
            # Modify this block to create a bar chart instead of scatter plot
            fig = {
                'data': [go.Bar(
                    x=df[xaxis_column_name],
                    y=df[yaxis_column_name],
                    text=df['App'],
                    marker={
                        'opacity': 0.7,
                        'line': {'width': 0.5, 'color': '#79E0EE'}
                    }
                )],
                'layout': go.Layout(
                    xaxis={'title': xaxis_column_name},
                    yaxis={'title': yaxis_column_name},
                    margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                    height=450,
                    hovermode='closest'
                )
            }
        elif xaxis_column_name == 'Android Ver':
            # Sort the dataframe by 'Android Ver'
            sorted_df = df.sort_values('Android Ver', key=lambda x: x.str.split().str[0])

            fig = {
                'data': [go.Box(
                    x=sorted_df[xaxis_column_name],
                    y=sorted_df[yaxis_column_name],
                    text=sorted_df['App'],
                )],
                'layout': go.Layout(
                    xaxis={'title': xaxis_column_name, 'categoryorder': 'array',
                           'categoryarray': sorted_df['Android Ver'].unique()},
                    yaxis={'title': yaxis_column_name},
                    margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                    height=450,
                    hovermode='closest'
                )
            }
        else:
            fig = {
                'data': [go.Box(
                    x=df[xaxis_column_name],
                    y=df[yaxis_column_name],
                    text=df['App'],
                )],
                'layout': go.Layout(
                    xaxis={'title': xaxis_column_name},
                    yaxis={'title': yaxis_column_name},
                    margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                    height=450,
                    hovermode='closest'
                )
            }
    return fig

@app.callback(
    dash.dependencies.Output('x-histogram', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_x_histogram(xaxis_column_name, xaxis_type):
    if xaxis_column_name == 'Last Updated':
        # Create a histogram for 'Last Updated' column
        fig = {
            'data': [go.Histogram(
                x=df[xaxis_column_name],
                marker=dict(color='#428bca')
            )],
            'layout': go.Layout(
                xaxis={'title': xaxis_column_name},
                yaxis={'title': 'Count'},
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=250,
                hovermode='closest'
            )
        }
    elif xaxis_column_name in ['Installs', 'Reviews', 'Rating', 'Size', 'Price', 'Current Ver']:
        fig = {
            'data': [go.Histogram(
                x=df[xaxis_column_name],
                # xbins={'start': df[xaxis_column_name].min(), 'end': df[xaxis_column_name].max(), 'size': 1},
            )],
            'layout': go.Layout(
                title=xaxis_column_name,
                xaxis={'type': 'log' if xaxis_type == 'Log' else 'linear'},
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=300,
                hovermode='closest'
            )
        }
    else:
        if xaxis_column_name in ['Last Updated', 'Genres', 'Android Ver']:  # Update this condition based on your specific column name
            # Count the occurrences of each category
            category_counts = df[xaxis_column_name].value_counts()

            # Select top 10 categories and group the rest as 'Others'
            top_categories = category_counts[:10]
            other_count = category_counts[10:].sum()

            # Create labels and values for the pie chart
            if other_count!=0:
                labels = list(top_categories.index) + ['Others']
                values = list(top_categories.values) + [other_count]

            fig = {
                'data': [go.Pie(
                    labels=labels,
                    values=values,
                )],
                'layout': go.Layout(
                    title=xaxis_column_name,
                    height=300,
                )
            }
        else:
            fig = {
                'data': [go.Pie(
                    labels=df[xaxis_column_name].value_counts().index,
                    values=df[xaxis_column_name].value_counts().values,
                )],
                'layout': go.Layout(
                    title=xaxis_column_name,
                    height=300,
                )
            }
    return fig

@app.callback(
    dash.dependencies.Output('y-histogram', 'figure'),
    [dash.dependencies.Input('crossfilter-yaxis-column', 'value')])

def update_y_histogram(yaxis_column_name):
    return {
        'data': [go.Histogram(
            x=df[yaxis_column_name]
        )],
        'layout': go.Layout(
            title=yaxis_column_name,
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=300,
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server()
