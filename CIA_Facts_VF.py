import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests  # Import the requests library to make HTTP requests
import openai
import plotly.express as px
import random
import time
import pickle
import sqlite3 
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor


#Loading Media
image1=Image.open('image1.png')

# DB Mgmt
import sqlite3 
#conn = sqlite3.connect('data/CIA_facts_cleaned.sqlite')
#c = conn.cursor()


# Fxn Make Execution
def sql_executor(raw_code):
    c.execute(raw_code)
    data = c.fetchall()
    return data 


CIA_facts_cleaned = ['Country,', 'Region,', 'Population,', 'Area (sq. mi.),', 'Pop. Density (per sq. mi.),', 'Coastline (coast/area ratio),', 'Net migration,', 'Infant mortality (per 1000 births),', 'GDP ($ per capita),', 'Literacy (%),', 'Phones (per 1000),', 'Arable (%),', 'Crops (%),', 'Other (%),', 'Climate,', 'Birthrate,', 'Deathrate,', 'Agriculture,', 'Industry,', 'Service']

def custom_sql_query():
    st.title("CIA Country Facts - Custom SQL Query")
    
    # Columns/Layout
    col1, col2 = st.columns(2)

    with col1:
        with st.form(key='query_form'):
            raw_code = st.text_area("SQL Code Here")
            submit_code = st.form_submit_button("Execute")

        # Table of Info
        with st.expander("Table Info"):
            table_info = {'CIA_cleaned_data': CIA_facts_cleaned}
            st.json(table_info)

    # Results Layouts
    with col2:
        if submit_code:
            st.info("Query Submitted")
            st.code(raw_code)

            # Results 
            query_results = sql_executor(raw_code)
            with st.expander("Results"):
                st.write(query_results)

            with st.expander("Pretty Table"):
                query_df = pd.DataFrame(query_results)
                st.dataframe(query_df)

# Data cleaning function
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('CIA_Country_Facts.csv')
    df.drop(columns='Climate', inplace=True)
    df_cleaned = df.dropna().copy()

    # Relabel countries
    country_name_mapping = {
        'Turks & Caicos Is': 'Turks & Caicos',
        'Saint Vincent and the Grenadines': 'The Grenadines',
        'St Pierre & Miquelon': 'St Pierre',
        'Korea, South': 'South Korea',
        'Saint Kitts & Nevis': 'Saint Kitts',
        'N. Mariana Islands': 'North Mariana Islands',
        'Korea, North': 'North Korea',
        'Micronesia, Fed. St.': 'Micronesia',
        'Gambia, The': 'Gambia',
        'Congo, Dem. Rep.': 'Democratic Republic of Congo',
        'Congo, Repub. of the': 'Congo',
        'Central African Rep.': 'Central African Republic',
        'British Virgin Is.': 'British Virgin Islands',
        'Bahamas, The': 'Bahamas',
        'Antigua & Barbuda': 'Antigua'
    }
    df_cleaned['Country'] = df_cleaned['Country'].replace(country_name_mapping)

    # Relabel regions
    region_name_mapping = {
        'ASIA (EX. NEAR EAST)': 'ASIA',
        'C.W. OF IND. STATES': 'INDEPENDENT STATES',
        'LATIN AMER. & CARIB': 'LATIN AMERICA',
        'NEAR EAST': 'MIDDLE EAST'
    }
    df_cleaned['Region'] = df_cleaned['Region'].str.strip().replace(region_name_mapping)
    return df_cleaned



# ADDED for ML MODEL
def load_model():
    """Load the serialized model."""
    with open('rf_model_for_gdp_prediction.pkl', 'rb') as model_file:
        return pickle.load(model_file)


# Function to generate facts using GPT API
def generate_facts(country, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'system', 'content': 'You are a CIA Analyst.'},
                     {'role': 'user', 'content': f'Please give a brief synopsis of the current situation in {country}. Be concise and not not use more than 5 lines'}],
        'max_tokens': 150,
        'temperature': 0.7  
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    if response.status_code == 200:
        facts = response.json()['choices'][0]['message']['content'].strip()  
        return facts
    else:
        st.write(f'Error: {response.status_code}')
        st.write(response.text)  # This will output the error message from the API

 
def sidebar_content(df):
    st.sidebar.header('Navigation')
    #selection = st.sidebar.selectbox("Go to", ["Home", "Comparative Analysis", "Country Insights", "Predictions", "Custom SQL Queries","About"])
    selection = st.sidebar.selectbox("Go to", ["Home", "Comparative Analysis", "Predictions", "Custom SQL Queries","About"])

    if selection == "Country Insights":
        country_selected = st.sidebar.selectbox("Select Country", df['Country'].unique())
    else:
        country_selected = None
    

    #question = st.sidebar.text_input('Ask a Question:')
    #ask_button = st.sidebar.button('Ask')

   
    #return selection, country_selected,  question, ask_button
    return selection, country_selected


def main_content_area(selection, country_selected, df): 
#def main_content_area(selection, country_selected, df, question, ask_button): 
                    
    # Call the function to get the cleaned dataframe
    
    df_cleaned = load_and_clean_data()
    #st.image(image1)
    if selection == "Home":
        st.write("## Introduction")
        st.write("Welcome to the CIA Country Facts App! Dive into a world of data and insights from countries across the globe. Whether you're a student, researcher, or just curious, this is your gateway to understanding nations better. Explore statistics, visualize trends, interact with maps, and even ask questions. Embark on this enlightening journey with us!")
        # DR: Indented bullets
        # Quick Statistics
        st.subheader("Quick Statistics")
        st.markdown(f"- **Total Countries:** {df_cleaned['Country'].nunique()}")
        st.markdown(f"- **Avg. GDP:** ${df_cleaned['GDP ($ per capita)'].mean():.2f}")
        st.markdown(f"- **Avg. Population:** {df_cleaned['Population'].mean():,.0f} people")
        st.markdown(f"- **Avg. Literacy Rate:** {df_cleaned['Literacy (%)'].mean():.2f}%")
                    
        st.write("-----")  # Add a separator        

        # Introduction and instructions for the search feature
        st.write("#### Explore by Country")
        st.write("""
        Input the name of any country in the text box below to get a quick preview of its flag, map, and key statistics.
        This feature is designed to help you quickly identify and get a visual sense of any nation. The autocomplete feature
        will assist you, but don't worry if you make a typo; we'll try our best to find a match for you!
        """)

        # Search for a country
        country_input = st.text_input("Search for a Country","Type Here")
        if country_input:
            filtered_countries = df_cleaned[df_cleaned['Country'].str.contains(country_input, case=False)]

            if not filtered_countries.empty:
                country_selected = filtered_countries['Country'].iloc[0]
                flag_url = filtered_countries['Flag'].iloc[0]
                map_url = filtered_countries['Map'].iloc[0]
                st.markdown(f"##### Preview of {country_selected}'s Flag and Location!")
                st.image(flag_url)  # Display the flag
                st.image(map_url)   # Display the map
                
                with st.expander("Table"):
                    st.write(f"**Country Details:**")
                    st.write("This table provides a comprehensive set of details for the selected country, ranging from demographic data to economic indicators.")
                    st.table(filtered_countries.T)  # Transpose the dataframe for better visibility
                    
                with st.expander("Charts"):
                    st.write(f"**Charts for {country_selected}**")
                    st.write("Explore visual representations of key metrics. Here, we present the distribution of GDP contributions across major sectors.")

                    # Sector-wise Contribution to GDP
                    sectors = ['Agriculture', 'Industry', 'Service']
                    contributions = [filtered_countries['Agriculture'].iloc[0], 
                                     filtered_countries['Industry'].iloc[0], 
                                     filtered_countries['Service'].iloc[0]]

                    
                    colors =['#FFCDB2', '#FFA69E', '#FAF3DD'] 

                    # Create a pie chart using matplotlib with adjusted size
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.pie(contributions, labels=sectors, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig2)
                    st.write("Sector-wise Contribution to GDP")


                with st.expander("Current Facts"):
                    st.write(f"**Current Situation in {country_selected}**")
                    st.write("Dive deeper into the present state of affairs in the selected country. This section fetches real-time insights using an external AI model, so it might take a moment to load. Your patience is appreciated.")
                    with open('key.txt', 'r') as file:
                        api_key = file.read().strip()
                    if st.button('More Facts'):
                        facts = generate_facts(country_selected, api_key)
                        st.info(facts)

            else:
                st.info("Input your country here!")


        st.write("-----")  # Add a separator
        
        # information for the Top 5 countries by GDP
        st.write("#### Economic Powerhouses")
        st.write("""
        Below is a list showcasing the top 5 countries based on their Gross Domestic Product (GDP) per capita. 
        GDP per capita is a measure of a country's economic performance, representing the average economic output per person if a country's total production is evenly divided among its citizens. 
        These nations stand out for their economic achievements and are often seen as benchmarks in the global economy.
        """)
            
        # Top 5 countries by GDP    
        st.write("**Top 5 Countries by GDP**")
        top_gdp = df_cleaned.nlargest(5, 'GDP ($ per capita)')
        for rank, (country, gdp) in enumerate(zip(top_gdp['Country'], top_gdp['GDP ($ per capita)']), start=1):
            st.markdown(f"{rank}. **{country}**: ${gdp:.2f}")
            
    
        st.write("-----")  # Add a separator
        st.markdown("""
        #### Top 10 Countries Chart
        Explore the top 10 countries based on different metrics available in the dataset. Simply choose a metric from the dropdown menu below and the chart will automatically update to display the top 10 countries based on your selection. Analyze the results and make your conclusions!
        """)

        # User selection for the variable they want to see the top 10 of
        available_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
        selected_column = st.selectbox("Select a metric:", available_columns)

        # Create the chart based on the user's selection
        top_10_data = df_cleaned.nlargest(10, selected_column)
        fig = px.bar(top_10_data, x='Country', y=selected_column, title=f"Top 10 Countries by {selected_column}", color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig)

        st.write("-----")  # Add a separator   

        # Random Selection
        
        st.subheader("Did You Know?")
        st.markdown("""
        Take a chance and learn something unexpected! By clicking the button below, you'll be presented with a random fact about a country from the dataset. Dive into the diverse world of countries and discover intriguing tidbits you might not have known before. Who knows? You might just stumble upon an interesting piece of information about a country you're not familiar with. Give it a try!
        """)
        if st.button("Generate Random Fact"):
            random_country = df_cleaned.sample(1).iloc[0]
            st.info(f"**{random_country['Country']}** has a population of {random_country['Population']:,}, a GDP of ${random_country['GDP ($ per capita)']:.2f}, and an infant mortality rate of {random_country['Infant mortality (per 1000 births)']} per 1,000 births.")

            
        st.write("-----")  # Add a separator  
        
        st.subheader("Feedback/Questions")
        st.markdown("""
        We value your input! Whether it's a suggestion, a question about the data, or just general feedback, we'd love to hear from you. This app is designed to be as informative and user-friendly as possible, and your insights can help make it even better. Simply type your thoughts in the box below and click 'Submit'. Thank you for helping us improve!
        """)
        feedback = st.text_area("Let us know your thoughts!")
        if st.button("Submit"):
            st.success("Thank you for your feedback!")

        # # API results
        # if ask_button:
        #     st.write("API response for:", question)
        #     st.write("Sample API response here...") 

        # DR: Updated API results
        # if ask_button:
        #     st.write("You asked:", question)
        #     with open('key.txt', 'r') as file:
        #         api_key = file.read().strip()
        #     answer = generate_facts(question, api_key)
        #     st.write("Answer:", answer)

                    
    elif selection == "Comparative Analysis":
        st.write("## Comparative Analysis")
        st.write("Welcome to the Comparative Analysis section. Here, you can explore and compare data across different countries and regions, shedding light on trends and insights.")

        categorical_columns = ['Country', 'Region']
        numerical_columns = [col for col in df.columns if col not in categorical_columns + ['Flag', 'Map']]

        # Overview: Distribution of regions
        st.write("### Dataset Overview")
        st.write("Before diving deeper, let's get a visual representation of the data distribution across regions.")
        fig2 = px.pie(df, names='Region', title='Distribution of Regions')
        st.plotly_chart(fig2)

        st.write("\n")

        # Feature to display
        st.write("### Choose Features to Analyze")
        st.write("Select from a list of numerical features to customize your analysis. Multiple selections are possible.")
        important_features = ['Population', 'Area (sq. mi.)', 'GDP ($ per capita)', 'Literacy (%)']
        feature = st.multiselect('Select features to analyze:', numerical_columns, default=important_features)
        if not feature:
            st.warning("Please select at least one feature to analyze.")
            return

        st.write("\n")

        # Visualization Type
        st.write("### Select Visualization Type")
        st.write("Different visualizations offer varying insights. Choose one based on your analytical needs:")
        chart_type = st.selectbox('Select visualization type:', ['Bar Chart', 'Violin Plot', 'Boxplot', 'Correlation Matrix'])

        st.write("\n")

        # Comparison Basis: Only Region
        comparison_basis = 'Region'
        st.write("### Select Your Region of Interest")
        st.write("For a granular analysis, narrow down to a specific region. This will filter the data to display results for the selected region only.")
        selected_regions = st.selectbox('Select Region:', df['Region'].unique())

        # Filtering the dataframe based on selections
        filtered_df = df.copy()
        if selected_regions:
            filtered_df = filtered_df[filtered_df[comparison_basis] == selected_regions]


        # Check if filtered data is empty
        if filtered_df.empty:
            st.warning("No data available for the selected criteria. Please adjust your selections.")
        else:
            st.write("### Visualization Results")
            st.write("Based on your selections, here's the visualization that represents the insights drawn from the data:")

            # Visualization Display
            if chart_type == 'Bar Chart':
                fig = px.bar(filtered_df, x=comparison_basis, y=feature[0], title=f'{feature[0]} by {comparison_basis}')
            elif chart_type == 'Violin Plot':
                fig = px.violin(filtered_df, y=feature[0], box=True, points="all", title=f'Distribution of {feature[0]} by {comparison_basis}')
            elif chart_type == 'Boxplot':
                fig = px.box(filtered_df, x=comparison_basis, y=feature[0], title=f'Distribution of {feature[0]} by {comparison_basis}')
            elif chart_type == 'Correlation Matrix':
                correlation_matrix = filtered_df[feature].corr()
                fig = px.imshow(correlation_matrix, title='Feature Correlation Matrix', labels=dict(color="Correlation Coefficient"))

            st.plotly_chart(fig)

            
            if chart_type == 'Correlation Matrix':
                st.write("Note: Correlation values range between -1 and 1. A value closer to 1 indicates a strong positive correlation, while a value closer to -1 indicates a strong negative correlation.")
                
        st.write("-----")  # Add a separator  
        
        st.write("\n")
        # Feature Distributions Across All Regions
        st.write("### Feature Distributions Across All Regions")
        st.write("Get an overview of the distribution of a particular feature across all regions in a single view.")
        distribution_feature = st.selectbox('Select a feature for distribution analysis:', numerical_columns)
        selected_regions_for_distribution = st.multiselect('Select regions to compare:', df['Region'].unique(), default=df['Region'].unique())
        # If no regions are selected, show a warning and stop further execution
        if not selected_regions_for_distribution:
            st.warning("Please select at least one region for distribution analysis.")
            return
        distribution_type = st.selectbox('Choose distribution visualization type:', ['Boxplot', 'Violin Plot'])

        # Filter data based on the regions selected
        filtered_df_for_distribution = df[df['Region'].isin(selected_regions_for_distribution)]

        if distribution_type == 'Boxplot':
            fig_distribution = px.box(filtered_df_for_distribution, x='Region', y=distribution_feature, title=f'Distribution of {distribution_feature} Across Selected Regions')
        elif distribution_type == 'Violin Plot':
            fig_distribution = px.violin(filtered_df_for_distribution, x='Region', y=distribution_feature, box=True, points="all", title=f'Distribution of {distribution_feature} Across Selected Regions')

        st.plotly_chart(fig_distribution)
        
        st.write("-----")  # Add a separator  
        st.write("\n")
        st.write("### Feature Interactions & Relationships")
        st.write("Explore how two features interact with each other. Gain insights into the relationships and trends among different features.")

        # Selecting two features
        feature_x = st.selectbox('Select Feature on X-axis:', numerical_columns, index=0)
        feature_y = st.selectbox('Select Feature on Y-axis:', numerical_columns, index=1)

        # Scatter plot without a trendline
        scatter_fig = px.scatter(df, x=feature_x, y=feature_y, title=f'Relationship between {feature_x} and {feature_y}')
        st.plotly_chart(scatter_fig)


        st.write("-----")  # Add a separator  

        st.write("\n")

        # Ranked Feature Comparison
        st.write("### Ranked Feature Comparison")
        st.write("View the top regions based on a selected feature.")

        ranked_feature = st.selectbox('Select a feature to rank:', numerical_columns)
        top_n = st.slider("Select the number of top countries/regions to display:", 2, 10, 5)

        if 'Region' not in df.columns or ranked_feature not in df.columns:
            st.error("The expected columns are not present in the dataframe.")
            return

        # Remove duplicates, if any, based on 'Region'
        df = df.drop_duplicates(subset=['Region'], keep='first')

        # Sort the data and get top_n values
        sorted_df = df.sort_values(by=ranked_feature, ascending=False).head(top_n)

        # Create the bar chart
        fig_ranked = px.bar(sorted_df, x='Region', y=ranked_feature, title=f'Top {top_n} Regions by {ranked_feature}')
        st.plotly_chart(fig_ranked)
            

    #elif selection == "Country Insights":
    #    st.write(f"## Insights for {country_selected}")
    #    country_data = df[df['Country'] == country_selected]
    #    st.table(country_data)  
        
        #Commented out per Ana
        # with st.container():
        #     st.subheader('Select the Plot you Want to See')
        #     plot=('box','violin','kdeplot','histogram')
        #     plot=st.selectbox('Select your chart',plot)
        #     fig,ax=plt.subplots()

        #     if plot=='box':
        #         sns.boxplot(x='Country',y='Population',data=df,ax=ax)
        #     elif plot=='violin':
        #         sns.violinplot(x='Country',y='GDP ($ per capita)',data=df,ax=ax)
        #     elif plot=='kdeplot':
        #         sns.kdeplot(x='Country',hue='Birthrate',data=df,ax=ax)
        #     else:
        #         sns.histplot(x='Country', hue='Net migration',data=df,ax=ax)

        #     st.pyplot(fig)

#     import geopandas as gpd

# # Load a world map shapefile
#     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# # Create a function to plot the map
#     def plot_country_map(country_name):
#         country = world[world['name'] == country_name]
#         fig = px.choropleth(country, geojson=country.geometry, locations=country.index, title=f"Map of {country_name}")
#         st.plotly_chart(fig)

# # In the "Country Insights" section, use the function to plot the map
#     if selection == "Country Insights":
#         #st.write(f"## Map of {country_selected}")
#     # ... (previous code)
# s
#     # Plot the map for the selected country
#         #st.subheader(f"Map of {country_selected}")
#         plot_country_map(country_selected)
    



    if selection == "Predictions":
        st.write("## Predictions")
        # DR: Updated these lines for clarity
        st.write("### Please make selections and then select Predict GDP per Capita ")
        st.write("These selections will be used as inputs to our Machine Learning model to predict GDP per Capita.  Based on our analysis, we determined these variables to be the most relevant to our predictive model. ")

        # Load the trained model
        model = load_model()

        phones = st.slider('Phones (per 1000)', int(df['Phones (per 1000)'].min()), int(df['Phones (per 1000)'].max()), int(df['Phones (per 1000)'].mean()))
        service = st.slider('Service', float(df['Service'].min()), float(df['Service'].max()), float(df['Service'].mean()))
        literacy = st.slider('Literacy (%)', float(df['Literacy (%)'].min()), float(df['Literacy (%)'].max()), float(df['Literacy (%)'].mean()))
        net_migration = st.slider('Net Migration', float(df['Net migration'].min()), float(df['Net migration'].max()), float(df['Net migration'].mean()))
        agriculture = st.slider('Agriculture', float(df['Agriculture'].min()), float(df['Agriculture'].max()), float(df['Agriculture'].mean()))
        infant_mortality = st.slider('Infant Mortality (per 1000 births)', float(df['Infant mortality (per 1000 births)'].min()), float(df['Infant mortality (per 1000 births)'].max()), float(df['Infant mortality (per 1000 births)'].mean()))
        birthrate = st.slider('Birthrate', float(df['Birthrate'].min()), float(df['Birthrate'].max()), float(df['Birthrate'].mean()))


        # Predict button
        if st.button("Predict GDP per Capita"):
            # Create a dataframe from the inputs
            input_data = pd.DataFrame({
                'Phones (per 1000)': [phones],
                'Service': [service],
                'Literacy (%)': [literacy],
                'Net migration': [net_migration],
                'Agriculture': [agriculture],
                'Infant mortality (per 1000 births)': [infant_mortality],
                'Birthrate': [birthrate]
            })
            
            # Make predictions
            prediction = model.predict(input_data)
            st.write(f"Predicted GDP per Capita: ${prediction[0]:,.2f}")
            
            
    elif selection == "Custom SQL Queries":
        st.write("This section allows you to execute custom SQL queries on the data. Please provide your SQL query in the input box and press 'Execute' to see the results.")
                   
        # SQL results
        query = st.sidebar.text_input('SQL Query:', 'SELECT * FROM data LIMIT 5')
        execute_sql = st.sidebar.button('Execute')
        if execute_sql:
            try:
                with sqlite3.connect('data.db') as conn:
                    results = pd.read_sql(query, conn)
                st.write(results)
            except Exception as e:
                st.write("Error executing query: ", e) 

    # DR:  Added this section
    elif selection == "About":
        st.write("## Features")
    
        markdown_content = """

                        ### Navigation Sidebar:
                        - Allows users to switch between different sections: Home, Comparative Analysis, Predictions, Custom SQL Queries, and About.
                        - Provides a country selection dropdown for specific country insights.

                        ### Home Section:
                        - Displays quick statistics on the total number of countries, average GDP, population, and literacy rate.
                        - Enables users to search for a country and view its flag, map, key statistics, a pie chart of GDP contributions, and current situation insights (Using Open AI / Chat GPT).
                        - When a country is selected, under Current Facts - there is an option to select "More Facts."  This utilizes an API call to Open AI to generate this result. 
                        - Offers an economic powerhouse section showcasing the top 5 countries based on GDP per capita.
                        - Visualizes the top 10 countries based on user-selected metrics.
                        - Delivers random facts about a random country.
                        - Includes a feedback/question section for user input.

                        ### Comparative Analysis Section:
                        - Allows users to compare data across different countries and regions.
                        - Provides various visualizations like pie charts, bar charts, violin plots, box plots, and scatter plots based on user-selected features.

                        ### Predictions Section:
                        - Provides a GDP per capita prediction feature based on user input for several factors (e.g., Phones per 1000, Service, Literacy).
                        - Utilizes a pre-trained Random Forest model.

                        ### Custom SQL Queries Section:
                        - Allows users to input and execute custom SQL queries on the dataset.

                        ## Technical Aspects:
                        - **Data Source**: The main data source is the CIA_Country_Facts.csv file, which is loaded and cleaned up within the application.
                        - **Libraries**: Utilizes libraries like pandas for data manipulation, Matplotlib and Seaborn for static visualizations, Plotly for interactive visualizations, SQLite for database operations, and scikit-learn for machine learning.
                        - **External API Calls**: Uses the OpenAI API to fetch real-time insights about countries.
                        - **Database**: Uses SQLite for database management, although the database connection setup is commented out.
                        - **Machine Learning**: Employs a Random Forest model (rf_model_for_gdp_prediction.pkl) for predicting GDP per capita based on various country statistics.
                        """
                        
        st.markdown(markdown_content)



def main():
    st.set_page_config(page_title="CIA Facts", layout="wide")
    st.title("CIA Country Facts")

    # Load and clean data
    df_cleaned = load_and_clean_data()


    # Sidebar
    #selection, country_selected, question, ask_button = sidebar_content(df_cleaned)
    selection, country_selected = sidebar_content(df_cleaned)

    
    # Main Content
    #main_content_area(selection, country_selected, df_cleaned, question, ask_button)
    main_content_area(selection, country_selected, df_cleaned)

if __name__ == "__main__":
    main()

    
    
    