# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 20:38:34 2021

@author: kvoul
"""

#Necessary Modules######
#To create the app
import streamlit as st
#To connect alpha-2 country codes to country names and vice versa
import pycountry

#A global variable to save on API calls
#in case it's true, news are loaded from a local file on my computer
on_test = False

def sentimental_news(my_api_key,category = "business", country = "-"):

    """
    Function that conducts: 
    queries towards the NewsApi
    sentiment analysis on the news returned from the call
    And returns a tidy dataframe.
    """

    import pandas as pd #To handle dataframes
    from newsapi import NewsApiClient #To handle news queries
    import nltk #For natural language processing
    import re #For regex
    from nltk.sentiment.vader import SentimentIntensityAnalyzer #For sentiment analysis
    
    #Coincides with Global
    if country == "-":
        country = None
    
    # Initialize object
    newsapi = NewsApiClient(api_key = my_api_key)
    
    #news query, tailor-made by user preferences
    news_query = newsapi.get_top_headlines(category = category, country = country)

    #In case no result found, return a string
    if news_query["totalResults"] == 0:
        return "No Headlines found for this Country in this News Category"
    
    def clean_tokenize(sentence, language = "english", remove_duplicates = False, tokenize = False):
        """
        Tokenizes a sentence and 
        
        Removes: 
        1. Punctuation,
        2. Stopwords,
        3. HTML elements and 
        3. Removes duplicate words (Optionally)
        
        Returns tokens in a list
        
        Args:
            sentence (str) : the sentence we intend to tokenize
            language (str): language as recorded in nltk stopwords corpus (24 languages as at Dec 2021)
            remove_duplicates(bool): whether we want duplicate words in the output 
            
        Returns:
            if tokenized - list: words of the sentence
            else str: clean sentence
        """
        if sentence is None:
            return ""
        
        def striphtml(data):
            """
            Removes html tags
            e.g. <title>, </title>, <head>, etc. 
            from a string
            """
            p = re.compile(r'<.*?>')
            return p.sub('', data)
        
        #Define stopwords - in this case english
        #nltk.download('stopwords') # First download corpus #not working on the online application hosted on streamlit
        #stop_words = (nltk.corpus.stopwords.words(language))
        
        #Instead, this will work:
        #Read lines from file included in the same directory
        with open('english_stopwords') as f: stop_words = f.readlines()
        #Remove the \n character
        stop_words = [word.replace("\n","") for word in stop_words]
      
        #Remove html tags
        sentence = striphtml(sentence)
        
        #Define punctuation
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        
        #Remove Punctuation
        for element in sentence:
            if element in punc:
                sentence = sentence.replace(element, "")

        #Remove stopwords and return
        nltk.download('punkt')
        to_be_returned = [ word.lower() for word in nltk.word_tokenize(sentence) if not word.lower() in stop_words ]

        #In case we don't want duplicate words
        if remove_duplicates:
            to_be_returned = [w for w in set(to_be_returned)]
        
        #In case we don't want words separated
        #THey will be returned as a sentence
        if not tokenize:
            to_be_returned = " ".join(to_be_returned)
        
        return to_be_returned
    
    def query_to_dataframe(news_query):
        """
        Custom - Made for the json response returned from the newsapi
        
        Args:
            news_query(dict): the response exactly as it returns from the API
        
        Returns:
            dataframe: all data structured in a table format
        
        """
        #Initiliaze all soon-to-be columns
        source_ids = []
        source_names = []
        authors = []
        titles = []
        titles_tokenized = []
        descriptions = []
        descriptions_tokenized = []
        urls = []
        publish_dates = []
        content = []
        content_tokenized = []
        images = []
        
        #For each article,
        #Insert each child node in respective list
        for article in news_query["articles"]:
            source_ids.append(article["source"]["id"])
            source_names.append(article["source"]["name"])
            authors.append(article["author"])
            titles.append(article["title"])
            titles_tokenized.append(clean_tokenize(article["title"]) )
            descriptions.append(article["description"])
            descriptions_tokenized.append(clean_tokenize(article["description"]))
            urls.append(article["url"])
            publish_dates.append(article["publishedAt"])
            content.append(article["content"])
            content_tokenized.append(clean_tokenize(article["content"]))
            images.append((article["urlToImage"]))
        
        #Input ready lists into a dataframe and return
        return pd.DataFrame( {"source_ids":source_ids,
                       "Source":source_names,
                       "authors":authors,
                       "titles":titles,
                       "titles_tokenized":titles_tokenized,
                       "Description":descriptions,
                       "descriptions_tokenized" : descriptions_tokenized,
                       "urls":urls,
                       "publish_dates":publish_dates,
                       "content":content,
                       "content_tokenized":content_tokenized,
                       "image_url":images} )
    
    #Turn the NEWSAPI response to a dataframe
    news_df = query_to_dataframe(news_query)
    
    # Sentiment Analysis ######
    #First, download appropriate lexicon
    nltk.downloader.download('vader_lexicon')
    #Initiate the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
    
    #What's the title's sentiment
    title_sentiment = news_df["titles_tokenized"].apply(analyzer.polarity_scores)
    #What's the description's sentiment
    description_sentiment = news_df["descriptions_tokenized"].apply(analyzer.polarity_scores)

    #Turn sentiments into columns
    news_df["Description_Sentiment"] = [desc["compound"] for desc in description_sentiment]
    news_df["Title_Sentiment"] = [tit["compound"] for tit in title_sentiment]
    #And calculate the mean sentiment of each new
    news_df["Final_Sentiment"] =  (news_df["Title_Sentiment"] + news_df["Description_Sentiment"]) / 2
    
    #Finally, sort from good to bad
    news_df = news_df.sort_values(by = "Final_Sentiment", ascending = False)

    return news_df

def make_clickable(link, content):
    #From areo:
    #https://discuss.streamlit.io/t/display-urls-in-dataframe-column-as-a-clickable-hyperlink/743/4
    return f'<a target="_blank" href="{link}">{content}</a>'

def show_image_from_url(image_url):
    return(f'<img src = {image_url} width = "250" height = "200" >')
    
#Global Variables##########
#All options that the NEWSapi function accepts 
Categories  = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
two_digit_countries = ['ae', 'ar', 'at', 'au', 'be', 'bg', 'br', 'ca', 'ch', 'cn', 'co', 'cu', 'cz', 'de', 'eg', 'es', 'fr', 'gb', 'gr', 'hk', 'hu', 'id', 'ie', 'il', 'in', 'is', 'it', 'jp', 'kr', 'lt', 'lv', 'ma', 'mx', 'my', 'ng', 'nl', 'no', 'nz', 'ph', 'pk', 'pl', 'pt', 'ro', 'rs', 'ru', 'sa', 'se', 'sg', 'si', 'sk', 'th', 'tr', 'tw', 'ua', 'us', 've', 'za']
whole_countries =  {pycountry.countries.get(alpha_2=country).name:country  for country in two_digit_countries} 
whole_countries["Global"] = "-"


def run():
    
    #Get Session's Global variables from user
    api_key = st.sidebar.text_input("Your NEWS API key") #The API Key
    category_name = st.sidebar.selectbox("Choose a News Category", [c.capitalize() for c in Categories], 2).lower() #The news category
    wanted_country = st.sidebar.selectbox("Choose a specific Country",list(whole_countries.keys()), (len(whole_countries.keys())-1) ) #The country from where news derive
    go_button = st.sidebar.button("▶ Make my Day, Please") #A button to initiate the whole process
    
    #Default page
    if not go_button:
        #Application's Description
        st.write("# 👋 Welcome to GoodNewsFirst!")
        st.markdown("GoodNewsFirst is a simple **application** aiming to bring you the latest **headlines** sorted by their **sentiment**. This means that for each news category you choose:")
        st.write("😁 The first new you get to see is the most positive ✅")
        st.write("😫 And the last one is the most negative❌")
        st.write("It Combines: ")
        st.write("📰 [NewsAPI](https://newsapi.org/)'s solution to search news woldwide.")
        st.write("🧪 [NLTK](https://www.nltk.org/)'s [SentimentIntensityAnalyzer](https://www.nltk.org/api/nltk.sentiment.sentiment_analyzer.html) to detect whether each new has a postive or negative spirit.")
        st.write("⚙ [Streamlit](https://streamlit.io/) module to easily build and deploy an application")
        st.write("To use it, you first need to register for a free API key [here](https://newsapi.org/docs/get-started).")
        st.write("Then, **use the sidebar** to navigate through headlines.")
        
    else: #Whenever the go_button is pressed:
        
        if on_test: #In case we don't want to spend api calls
            import pandas as pd 
            #Load dataframe of news from my local path
            df = pd.read_csv("C:/Users/kvoul/Desktop/Projects/News/Temp_df.csv",delimiter = ",")
            
        else: #If not on test
            #Try fetching the news
            df = sentimental_news(my_api_key = api_key,category = category_name, country = whole_countries[wanted_country])       
        
        #Remove all previous html elements
        for i in range(10):
            st.empty()
    
        #Display News Category and Country
        st.markdown("# %s" % (category_name.capitalize() + " News of " + wanted_country) )
        
        #In case no news where found
        if isinstance(df,str):
            #Just inform user about it
            st.write(df)     

        else: #In case some news were returned from our query
            #Render title a hyperlink
            df['Title'] = df.apply( lambda x: make_clickable(x['urls'],x['titles']), axis = 1 )
            #Make image to show up in the app
            df['image'] = df.apply( lambda x: show_image_from_url(x['image_url']), axis = 1 )
            #Subset our large dataframe
            df = df[["Title","Description", "Source", "image"]]        
            #Turn dataframe into html
            df = df.to_html(escape=False, header = False, index = False )
            #Show results            
            st.write(df, unsafe_allow_html=True)

#Action (the only action taken throughout the whole script)
run()
