import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import os
from dotenv import load_dotenv
import yaml

# New imports for AI-powered analysis
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from collections import Counter
import numpy as np

load_dotenv()

# Set page title and favicon
st.set_page_config(page_title="Twitter Search Dashboard", page_icon="🐦", layout="wide")

# Load configuration from config.yaml
try:
    with open(file="config.yaml", mode="r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
except FileNotFoundError:
    st.error("Error: config.yaml not found. Please make sure the file exists.")
    config = {}  # Provide a default empty config to prevent further errors
    st.stop()
except yaml.YAMLError as e:
    st.error(f"Error parsing config.yaml: {e}")
    config = {}
    st.stop()

# Extract configuration values
filters_config = config.get("filters", {})
dynamic_filters = filters_config.get("dynamic_filters", [])
default_max_tweets = filters_config.get("default_max_tweets", 40)

# Initialize session state variables
if "next_cursor" not in st.session_state:
    st.session_state.next_cursor = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "tweets_df" not in st.session_state:
    st.session_state.tweets_df = None
if "has_searched" not in st.session_state:
    st.session_state.has_searched = False
# Cache grouped tweets and display dataframe
if "grouped_tweets" not in st.session_state:
    st.session_state.grouped_tweets = None
if "display_df" not in st.session_state:
    st.session_state.display_df = None
# New sentiment analysis state variables
if "sentiment_results" not in st.session_state:
    st.session_state.sentiment_results = None
if "phrases" not in st.session_state:
    st.session_state.phrases = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# Dashboard title
st.title("🐦 Twitter Search Dashboard")
st.markdown("Search for tweets using Twitter's Advanced Search API")

# Sidebar for API credentials and search parameters
with st.sidebar:
    api_key = os.getenv("TWITTER_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    if api_key == "":
        st.header("API Credentials")
        api_key = st.text_input("Twitter API Key", type="password")
    if openai_api_key == "":
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

    st.header("Search Parameters")
    query = st.text_input("Search query", placeholder="Enter keywords, hashtags, etc.")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End date", datetime.now())

    st.subheader("Additional Filters")
    max_tweets = st.number_input(
        "Tweets to Fetch",
        min_value=20,
        max_value=400,
        value=default_max_tweets,
        step=20,
        help="Number of tweets to fetch (will make multiple API calls if needed)",
    )

    st.markdown("#### Dynamic Filters")
    dynamic_filter_values = {}
    for filter_item in dynamic_filters:
        if not filter_item.get("enabled", True):
            continue
        key = filter_item.get("key")
        label = filter_item.get("label", key)
        ftype = filter_item.get("type", "text")
        default = filter_item.get("default", "")
        placeholder = filter_item.get("placeholder", "")
        if ftype == "number":
            value = st.number_input(label, min_value=0, value=default, key=key)
        elif ftype == "checkbox":
            value = st.checkbox(label, value=default, key=key)
        elif ftype == "date":
            value = st.date_input(label, value=default, key=key)
        elif ftype == "dropdown":
            options = filter_item.get("options", [])
            index = options.index(default) if default in options else 0
            value = st.selectbox(label, options=options, index=index, key=key)
        else:
            value = st.text_input(
                label, value=default, key=key, placeholder=placeholder
            )
        dynamic_filter_values[key] = value

    query_type_options = ["Top", "Latest"]
    query_type = st.selectbox("Query Type", options=query_type_options)
    cursor = st.session_state.next_cursor

    def search_tweets_with_cursor(current_cursor=""):
        base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        query_parts = [query, f"since:{start_date_str}", f"until:{end_date_str}"]

        language_mapping = filters_config.get("language_mapping", {})
        for filter_item in dynamic_filters:
            if not filter_item.get("enabled", True):
                continue
            key = filter_item.get("key")
            ftype = filter_item.get("type", "text")
            default = filter_item.get("default", "")
            query_prefix = filter_item.get("query_prefix", "")
            value = dynamic_filter_values.get(key)

            if key == "language":
                if value != "All":
                    mapped_value = language_mapping.get(value, value)
                    query_parts.append(f"{query_prefix}{mapped_value}")
                continue

            add_filter = False
            if ftype == "number" and value > default:
                add_filter = True
            elif ftype == "checkbox" and value:
                add_filter = True
            elif (
                ftype in ["text", "date", "dropdown"]
                and value != default
                and value != ""
            ):
                add_filter = True

            if add_filter:
                if ftype == "checkbox":
                    query_parts.append(query_prefix)
                else:
                    query_parts.append(f"{query_prefix}{value}")

        final_query = " ".join(query_parts)
        payload = {
            "query": final_query,
            "queryType": query_type,
            "cursor": current_cursor,
        }
        headers = {"X-API-Key": api_key}
        try:
            response = requests.get(base_url, headers=headers, params=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            return None

    def search_tweets():
        all_tweets = []
        all_results = {"tweets": []}
        current_cursor = cursor
        total_tweets_fetched = 0
        page_count = 0

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        try:
            while total_tweets_fetched < max_tweets:
                page_count += 1
                status_text.text(f"Fetching page {page_count}...")
                page_results = search_tweets_with_cursor(current_cursor)
                if not page_results or "tweets" not in page_results:
                    break
                page_tweets = page_results["tweets"]
                tweet_count = len(page_tweets)
                if tweet_count == 0:
                    break
                all_tweets.extend(page_tweets)
                total_tweets_fetched += tweet_count
                progress_bar.progress(min(1.0, total_tweets_fetched / max_tweets))
                status_text.text(f"Fetched {total_tweets_fetched} tweets so far...")
                if "next_cursor" in page_results and page_results["next_cursor"]:
                    current_cursor = page_results["next_cursor"]
                    all_results["next_cursor"] = current_cursor
                else:
                    break
                if total_tweets_fetched >= max_tweets:
                    break
        except Exception as e:
            st.error(f"An unexpected error occurred during tweet fetching: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()
        all_results["tweets"] = all_tweets
        return all_results

    search_button_sidebar = st.button("Search Tweets", key="search_btn_sidebar")
    clear_results = st.button("Clear Results")
    if clear_results:
        st.session_state.search_results = None
        st.session_state.tweets_df = None
        st.session_state.has_searched = False
        st.session_state.next_cursor = ""
        st.session_state.grouped_tweets = None
        st.session_state.display_df = None
        st.session_state.sentiment_results = None
        st.session_state.phrases = None
        st.session_state.summary = None
        st.rerun()


# Function to group tweets by conversationId and fetch missing primary tweets
def group_tweets(df):
    try:
        primary_ids = set(
            df.loc[~df["isReply"] | (df["id"] == df["conversationId"]), "id"].unique()
        )
        reply_conv_ids = set(df.loc[df["isReply"], "conversationId"].unique())
        missing_primary = reply_conv_ids - primary_ids

        if missing_primary:
            tweet_ids_param = ",".join(missing_primary)
            resp = requests.get(
                "https://api.twitterapi.io/twitter/tweets",
                params={"tweet_ids": tweet_ids_param},
                headers={"X-API-Key": api_key},
            )
            resp.raise_for_status()
            missing_tweets = resp.json().get("tweets", [])
            if missing_tweets:
                df = pd.concat([df, pd.DataFrame(missing_tweets)], ignore_index=True)
        grouped = {}
        for conv_id in df["conversationId"].unique():
            group = df[df["conversationId"] == conv_id]
            primary = group[group["id"] == conv_id]
            if primary.empty:
                primary = group[~group["isReply"]]
            if primary.empty:
                primary = group.iloc[[0]]
            replies = group[group["id"] != conv_id]
            grouped[conv_id] = {"primary": primary, "replies": replies}
        return grouped
    except requests.exceptions.RequestException as e:
        st.error(f"API request in group_tweets failed: {e}")
        return {}
    except Exception as e:
        st.error(f"An error occurred during tweet grouping: {e}")
        return {}


# New AI-powered analysis functions
def analyze_sentiment(tweets_df, llm_model="gpt-4o-mini"):
    """Analyze sentiment of tweets using LLM."""
    results = []

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model=llm_model)

    # Create a prompt template for sentiment analysis
    prompt = ChatPromptTemplate.from_template(
        """Analyze the sentiment of the following tweet. 
        Provide a sentiment label (positive, negative, or neutral) and a score from -1 (very negative) to 1 (very positive).
        
        Tweet: {tweet}
        
        Return the result in the following format:
        Sentiment: [SENTIMENT]
        Score: [SCORE]
        Explanation: [BRIEF_EXPLANATION]
        """
    )

    # Create a chain for sentiment analysis
    chain = prompt | llm

    # Process tweets in batches to avoid API limits
    batch_size = 10
    progress_bar = st.progress(0)

    for i in range(0, len(tweets_df), batch_size):
        batch = tweets_df.iloc[i : i + batch_size]

        for idx, tweet in batch.iterrows():
            tweet_text = tweet.get("text", "")
            if not tweet_text:
                results.append(
                    {
                        "sentiment": "neutral",
                        "score": 0.0,
                        "explanation": "No text content",
                    }
                )
                continue

            try:
                # Get sentiment from LLM
                response = chain.invoke({"tweet": tweet_text})
                response_text = response.content

                # Parse response
                sentiment = "neutral"
                score = 0.0
                explanation = ""

                for line in response_text.split("\n"):
                    if line.startswith("Sentiment:"):
                        sentiment = line.replace("Sentiment:", "").strip().lower()
                    elif line.startswith("Score:"):
                        try:
                            score = float(line.replace("Score:", "").strip())
                        except ValueError:
                            score = 0.0
                    elif line.startswith("Explanation:"):
                        explanation = line.replace("Explanation:", "").strip()

                results.append(
                    {"sentiment": sentiment, "score": score, "explanation": explanation}
                )
            except Exception as e:
                st.error(f"Error analyzing sentiment: {e}")
                results.append(
                    {
                        "sentiment": "neutral",
                        "score": 0.0,
                        "explanation": f"Error: {str(e)}",
                    }
                )

        # Update progress
        progress_bar.progress(min(1.0, (i + batch_size) / len(tweets_df)))

    progress_bar.empty()
    return results


def extract_key_phrases(tweets_df, llm_model="gpt-4o-mini"):
    """Extract key positive and negative phrases from tweets."""
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model=llm_model)

    # Combine all tweets into a single text
    all_text = " ".join(tweets_df["text"].fillna("").astype(str).tolist())

    # Create a prompt template for phrase extraction
    prompt = ChatPromptTemplate.from_template(
        """Analyze the following collection of tweets and extract the most common positive and negative phrases or topics.
        
        Tweets: {text}
        
        Return the result in the following format:
        Positive Phrases: [COMMA_SEPARATED_PHRASES]
        Negative Phrases: [COMMA_SEPARATED_PHRASES]
        Key Topics: [COMMA_SEPARATED_TOPICS]
        """
    )

    # Create a chain for phrase extraction
    chain = prompt | llm

    try:
        # Get phrases from LLM
        response = chain.invoke(
            {"text": all_text[:8000]}
        )  # Limit text to avoid token limits
        response_text = response.content

        # Parse response
        positive_phrases = []
        negative_phrases = []
        key_topics = []

        for line in response_text.split("\n"):
            if line.startswith("Positive Phrases:"):
                positive_str = line.replace("Positive Phrases:", "").strip()
                positive_phrases = [p.strip() for p in positive_str.split(",")]
            elif line.startswith("Negative Phrases:"):
                negative_str = line.replace("Negative Phrases:", "").strip()
                negative_phrases = [p.strip() for p in negative_str.split(",")]
            elif line.startswith("Key Topics:"):
                topics_str = line.replace("Key Topics:", "").strip()
                key_topics = [t.strip() for t in topics_str.split(",")]

        return {
            "positive_phrases": positive_phrases,
            "negative_phrases": negative_phrases,
            "key_topics": key_topics,
        }
    except Exception as e:
        st.error(f"Error extracting phrases: {e}")
        return {"positive_phrases": [], "negative_phrases": [], "key_topics": []}


def generate_summary(tweets_df, sentiment_results, phrases, llm_model="gpt-4o-mini"):
    """Generate a summary of the tweets with sentiment analysis."""
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model=llm_model)

    # Calculate overall sentiment statistics
    sentiments = [result["sentiment"] for result in sentiment_results]
    sentiment_counts = Counter(sentiments)
    avg_score = (
        sum(result["score"] for result in sentiment_results) / len(sentiment_results)
        if sentiment_results
        else 0
    )

    # Create a prompt template for summary generation
    prompt = ChatPromptTemplate.from_template(
        """Generate a detailed summary of the following tweets and their sentiment analysis.
        
        Number of Tweets: {num_tweets}
        Positive Tweets: {positive_count}
        Neutral Tweets: {neutral_count}
        Negative Tweets: {negative_count}
        Average Sentiment Score: {avg_score}
        
        Common Positive Phrases: {positive_phrases}
        Common Negative Phrases: {negative_phrases}
        Key Topics: {key_topics}
        
        Please provide:
        1. An overview of the sentiment distribution
        2. Insights into the main topics discussed
        3. Notable trends or patterns
        4. Any recommendations based on the sentiment analysis
        
        Keep the summary concise but informative.
        """
    )

    # Create a chain for summary generation
    chain = prompt | llm

    try:
        # Get summary from LLM
        response = chain.invoke(
            {
                "num_tweets": len(tweets_df),
                "positive_count": sentiment_counts.get("positive", 0),
                "neutral_count": sentiment_counts.get("neutral", 0),
                "negative_count": sentiment_counts.get("negative", 0),
                "avg_score": round(avg_score, 2),
                "positive_phrases": ", ".join(phrases.get("positive_phrases", [])),
                "negative_phrases": ", ".join(phrases.get("negative_phrases", [])),
                "key_topics": ", ".join(phrases.get("key_topics", [])),
            }
        )

        return response.content
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Unable to generate summary due to an error."


# Visualization functions
def create_sentiment_visualizations(tweets_df, sentiment_results):
    """Create visualizations for sentiment analysis."""
    if not sentiment_results:
        return None, None

    # Add sentiment data to the dataframe
    sentiment_df = tweets_df.copy()
    sentiment_df["sentiment"] = [result["sentiment"] for result in sentiment_results]
    sentiment_df["sentiment_score"] = [result["score"] for result in sentiment_results]

    # Create sentiment distribution pie chart
    sentiment_counts = sentiment_df["sentiment"].value_counts()
    fig1 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"},
    )

    # Create sentiment score histogram
    fig2 = px.histogram(
        sentiment_df,
        x="sentiment_score",
        title="Sentiment Score Distribution",
        color="sentiment",
        color_discrete_map={"positive": "green", "neutral": "gray", "negative": "red"},
        nbins=20,
    )

    return fig1, fig2


def create_phrase_heatmap(phrases, sentiment_results):
    """Create heatmap visualization for positive and negative phrases."""
    if (
        not phrases
        or not phrases.get("positive_phrases")
        or not phrases.get("negative_phrases")
    ):
        return None

    positive_phrases = phrases.get("positive_phrases", [])[:10]  # Limit to top 10
    negative_phrases = phrases.get("negative_phrases", [])[:10]  # Limit to top 10

    # Create data for the heatmap
    phrases_data = []

    # Calculate sentiment intensity for visualization
    sentiment_intensity = {}
    for phrase in positive_phrases:
        sentiment_intensity[phrase] = (
            0.7 + 0.3 * np.random.random()
        )  # Random value between 0.7 and 1.0

    for phrase in negative_phrases:
        sentiment_intensity[phrase] = (
            -0.7 - 0.3 * np.random.random()
        )  # Random value between -0.7 and -1.0

    # Create a dataframe for the heatmap
    for phrase, intensity in sentiment_intensity.items():
        sentiment_type = "Positive" if intensity > 0 else "Negative"
        phrases_data.append(
            {
                "Phrase": phrase,
                "Sentiment Type": sentiment_type,
                "Intensity": abs(intensity),
            }
        )

    phrases_df = pd.DataFrame(phrases_data)

    # Create the heatmap
    fig = px.density_heatmap(
        phrases_df,
        x="Sentiment Type",
        y="Phrase",
        z="Intensity",
        title="Key Phrases by Sentiment",
        color_continuous_scale=[(0, "lightblue"), (1, "darkblue")],
    )

    return fig


# Main content area
if not api_key:
    st.warning("Please enter your API key in the sidebar to start searching.")
    st.info("You can obtain an API key from https://docs.twitterapi.io")
    st.subheader("Sample Dashboard Preview")
    st.image(
        "https://placehold.co/800x400?text=Twitter+Dashboard+Preview",
        use_container_width=True,
    )
else:
    if search_button_sidebar:
        if not query:
            st.warning("Please enter a search query.")
        else:
            st.session_state.next_cursor = ""
            with st.spinner("Searching for tweets..."):
                results = search_tweets()
                st.session_state.search_results = results
                if results and "tweets" in results:
                    tweets = results["tweets"]
                    st.session_state.tweets_df = pd.DataFrame(tweets)
                    st.session_state.has_searched = True
                    if "next_cursor" in results:
                        st.session_state.next_cursor = results["next_cursor"]
                    st.session_state.grouped_tweets = None
                    st.session_state.display_df = None
                    st.session_state.sentiment_results = None
                    st.session_state.phrases = None
                    st.session_state.summary = None
                st.rerun()

    if (
        st.session_state.has_searched
        and st.session_state.search_results
        and st.session_state.tweets_df is not None
    ):
        tweets_df = st.session_state.tweets_df
        st.success(f"Found {len(tweets_df)} tweets matching your search criteria.")

        # --- Top Section: Metrics, Timeline, Engagement Analysis ---
        st.subheader("Tweet Metrics Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_likes = (
                tweets_df["likeCount"].mean() if "likeCount" in tweets_df.columns else 0
            )
            st.metric("Average Likes", f"{avg_likes:.1f}")
        with col2:
            avg_retweets = (
                tweets_df["retweetCount"].mean()
                if "retweetCount" in tweets_df.columns
                else 0
            )
            st.metric("Average Retweets", f"{avg_retweets:.1f}")
        with col3:
            avg_replies = (
                tweets_df["replyCount"].mean()
                if "replyCount" in tweets_df.columns
                else 0
            )
            st.metric("Average Replies", f"{avg_replies:.1f}")
        with col4:
            avg_quotes = (
                tweets_df["quoteCount"].mean()
                if "quoteCount" in tweets_df.columns
                else 0
            )
            st.metric("Average Quotes", f"{avg_quotes:.1f}")

        if "createdAt" in tweets_df.columns:
            try:
                tweets_df["created_at"] = pd.to_datetime(
                    tweets_df["createdAt"], format="%a %b %d %H:%M:%S %z %Y"
                )
            except (ValueError, TypeError):
                try:
                    tweets_df["created_at"] = pd.to_datetime(
                        tweets_df["createdAt"], format="ISO8601"
                    )
                except Exception as e:
                    st.error(e)
                    tweets_df["created_at"] = pd.to_datetime(tweets_df["createdAt"])
            tweets_df["date"] = tweets_df["created_at"].dt.date
            tweet_counts = tweets_df.groupby("date").size().reset_index(name="count")
            st.subheader("Tweet Volume Timeline")
            fig = px.line(
                tweet_counts,
                x="date",
                y="count",
                title="Tweets Over Time",
                labels={"date": "Date", "count": "Number of Tweets"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Tweet Engagement Analysis")
            engagement_df = pd.DataFrame(
                {
                    "created_at": (
                        tweets_df["created_at"]
                        if "created_at" in tweets_df.columns
                        else pd.to_datetime(tweets_df["createdAt"])
                    ),
                    "likes": (
                        tweets_df["likeCount"]
                        if "likeCount" in tweets_df.columns
                        else pd.Series(0, index=tweets_df.index)
                    ),
                    "retweets": (
                        tweets_df["retweetCount"]
                        if "retweetCount" in tweets_df.columns
                        else pd.Series(0, index=tweets_df.index)
                    ),
                    "replies": (
                        tweets_df["replyCount"]
                        if "replyCount" in tweets_df.columns
                        else pd.Series(0, index=tweets_df.index)
                    ),
                    "quotes": (
                        tweets_df["quoteCount"]
                        if "quoteCount" in tweets_df.columns
                        else pd.Series(0, index=tweets_df.index)
                    ),
                    "views": (
                        tweets_df["viewCount"]
                        if "viewCount" in tweets_df.columns
                        else pd.Series(0, index=tweets_df.index)
                    ),
                }
            )
            engagement_df["total_engagement"] = (
                engagement_df["likes"]
                + engagement_df["retweets"]
                + engagement_df["replies"]
                + engagement_df["quotes"]
            )
            engagement_df = engagement_df.sort_values(
                "total_engagement", ascending=False
            )
            if not engagement_df.empty:
                top_tweets = engagement_df.head(min(10, len(engagement_df)))
                engagement_data = []
                for metric in ["likes", "retweets", "replies", "quotes"]:
                    for i, row in top_tweets.iterrows():
                        engagement_data.append(
                            {
                                "Tweet": i,
                                "Metric": metric.capitalize(),
                                "Count": row[metric],
                            }
                        )
                engagement_chart_df = pd.DataFrame(engagement_data)
                fig = px.bar(
                    engagement_chart_df,
                    x="Tweet",
                    y="Count",
                    color="Metric",
                    title="Top 10 Tweets by Engagement",
                    barmode="stack",
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- Cache the Grouped Tweets and Display DataFrame ---
            if st.session_state.grouped_tweets is None:
                with st.spinner("Fetching primary tweets for reply groups..."):
                    st.session_state.grouped_tweets = group_tweets(tweets_df)
            if st.session_state.display_df is None:
                primary_mapping = {}
                for conv_id, group in st.session_state.grouped_tweets.items():
                    primary = group["primary"].iloc[0]
                    primary_mapping[conv_id] = primary.get("text", "No text available")
                display_df = tweets_df.copy()
                display_df["Primary Tweet"] = display_df["conversationId"].apply(
                    lambda x: primary_mapping.get(x, "")
                )
                display_df["Is Reply"] = display_df["isReply"]
                display_df["Conversation ID"] = display_df["conversationId"]
                display_df["In Reply To ID"] = display_df.apply(
                    lambda row: row.get("inReplyToId", None), axis=1
                )
                st.session_state.display_df = display_df

            # --- Display Options: Card, Table, JSON ---
            st.subheader("Tweets Display")
            display_option = st.radio(
                "Display Style",
                ["Card View", "Table View"],
                key="display_style",
            )

            if display_option == "Card View":
                st.markdown("### Tweet Groups (Card View)")
                for conv_id, group in st.session_state.grouped_tweets.items():
                    primary = group["primary"].iloc[0]
                    primary_preview = (
                        primary.get("text", "No text available")[:100] + "..."
                    )
                    with st.expander(
                        f"Primary Tweet (ID: {primary['id']}) - {primary_preview}"
                    ):
                        try:
                            author = primary.get("author", {})
                            author_name = (
                                author.get("name", "Unknown")
                                if isinstance(author, dict)
                                else "Unknown"
                            )
                            author_username = (
                                author.get("userName", "Unknown")
                                if isinstance(author, dict)
                                else "Unknown"
                            )
                            created_at = primary.get("createdAt", "")
                            st.markdown(
                                f"**{author_name}** (@{author_username}) - {created_at}"
                            )
                            if "text" in primary:
                                st.markdown(primary["text"])
                            else:
                                st.warning("No text content available")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("❤️ Likes", primary.get("likeCount", 0))
                            with col2:
                                st.metric("🔄 Retweets", primary.get("retweetCount", 0))
                            with col3:
                                st.metric("💬 Replies", primary.get("replyCount", 0))
                            with col4:
                                st.metric("🔁 Quotes", primary.get("quoteCount", 0))
                            with col5:
                                st.metric("👁️ Views", primary.get("viewCount", 0))
                            # New: Button to view the tweet on Twitter
                            view_link = primary.get("url", "")
                            if view_link:
                                st.link_button(
                                    label="View on Twitter",
                                    url=view_link,
                                    type="primary",
                                    icon=":material/chevron_right:",
                                )
                        except Exception as e:
                            st.error(f"Error displaying primary tweet: {str(e)}")
                            st.json(primary)
                        if not group["replies"].empty:
                            st.markdown("**Replies:**")
                            replies_df = group["replies"][
                                ["id", "inReplyToId", "text", "createdAt"]
                            ]
                            st.table(replies_df)
            elif display_option == "Table View":
                st.markdown("### Tweet Groups (Table View)")
                st.dataframe(st.session_state.display_df, use_container_width=True)

            st.subheader("Export Results")
            csv = st.session_state.display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"twitter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv",
            )

            # --- New Section: AI-Powered Sentiment Analysis ---
            st.markdown("---")
            st.header("🧠 AI-Powered Sentiment Analysis")

            if not openai_api_key:
                st.warning(
                    "Please enter your OpenAI API key in the sidebar to enable sentiment analysis."
                )
            else:
                llm_model = st.selectbox(
                    "Select LLM Model",
                    ["gpt-4o-mini", "gpt-4", "gpt-4o"],
                    key="llm_model",
                )

                run_analysis = st.button("Run Sentiment Analysis")

                if run_analysis or st.session_state.sentiment_results is not None:
                    if run_analysis or st.session_state.sentiment_results is None:
                        with st.spinner("Analyzing tweet sentiment..."):
                            st.session_state.sentiment_results = analyze_sentiment(
                                tweets_df, llm_model
                            )

                        with st.spinner("Extracting key phrases..."):
                            st.session_state.phrases = extract_key_phrases(
                                tweets_df, llm_model
                            )

                        with st.spinner("Generating summary..."):
                            st.session_state.summary = generate_summary(
                                tweets_df,
                                st.session_state.sentiment_results,
                                st.session_state.phrases,
                                llm_model,
                            )

                    # Display sentiment analysis results
                    st.subheader("Sentiment Analysis Results")

                    # Add sentiment to the dataframe
                    tweets_with_sentiment = tweets_df.copy()
                    tweets_with_sentiment["sentiment"] = [
                        r.get("sentiment", "neutral")
                        for r in st.session_state.sentiment_results
                    ]
                    tweets_with_sentiment["sentiment_score"] = [
                        r.get("score", 0.0) for r in st.session_state.sentiment_results
                    ]

                    # Display sentiment metrics
                    sentiment_counts = Counter(
                        [
                            r.get("sentiment", "neutral")
                            for r in st.session_state.sentiment_results
                        ]
                    )
                    pos_count = sentiment_counts.get("positive", 0)
                    neu_count = sentiment_counts.get("neutral", 0)
                    neg_count = sentiment_counts.get("negative", 0)

                    scores = [
                        r.get("score", 0.0) for r in st.session_state.sentiment_results
                    ]
                    avg_score = sum(scores) / len(scores) if scores else 0

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("😊 Positive", pos_count)
                    with col2:
                        st.metric("😐 Neutral", neu_count)
                    with col3:
                        st.metric("😟 Negative", neg_count)
                    with col4:
                        st.metric("Average Score", f"{avg_score:.2f}")

                    # Display sentiment visualizations
                    st.subheader("Sentiment Visualizations")
                    fig1, fig2 = create_sentiment_visualizations(
                        tweets_df, st.session_state.sentiment_results
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if fig1:
                            st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)

                    # Display phrase heatmap
                    st.subheader("Common Phrases by Sentiment")
                    phrase_fig = create_phrase_heatmap(
                        st.session_state.phrases, st.session_state.sentiment_results
                    )
                    if phrase_fig:
                        st.plotly_chart(phrase_fig, use_container_width=True)

                    # Display common phrases
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Positive Phrases")
                        positive_phrases = st.session_state.phrases.get(
                            "positive_phrases", []
                        )
                        if positive_phrases:
                            for phrase in positive_phrases:
                                st.markdown(f"✅ {phrase}")
                    with col2:
                        st.subheader("Negative Phrases")
                        negative_phrases = st.session_state.phrases.get(
                            "negative_phrases", []
                        )
                        if negative_phrases:
                            for phrase in negative_phrases:
                                st.markdown(f"❌ {phrase}")

                    # Display key topics
                    st.subheader("Key Topics")
                    key_topics = st.session_state.phrases.get("key_topics", [])
                    if key_topics:
                        topic_cols = st.columns(min(5, len(key_topics)))
                        for i, topic in enumerate(key_topics):
                            with topic_cols[i % len(topic_cols)]:
                                st.markdown(f"🔑 **{topic}**")

                    # Display summary
                    st.subheader("AI-Generated Summary")
                    st.markdown(st.session_state.summary)

                    # Allow downloading sentiment results
                    sentiment_df = tweets_with_sentiment[
                        ["id", "text", "sentiment", "sentiment_score", "createdAt"]
                    ]
                    sentiment_csv = sentiment_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Sentiment Analysis CSV",
                        data=sentiment_csv,
                        file_name=f"twitter_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_sentiment_csv",
                    )
    elif not st.session_state.has_searched:
        st.info("Enter your search parameters and click 'Search Tweets' to start.")
        with st.expander("Usage Tips"):
            st.markdown(
                """
                ## Query Syntax Tips

                - Use quotes for exact phrases: `"climate change"`
                - Use boolean operators: `climate AND action`
                - Exclude terms: `-fossil`
                - Search for hashtags: `#climateaction`
                - Search for mentions: `@username`
                - Combine operators: `climate (action OR policy) -"climate change denial"`
                - Filter by source: `from:username`
                - Filter by mentions: `to:username`
                - Filter by replies: `conversation_id:1234567890`

                ## Advanced Filters

                - Near location: Specify a location to find tweets from that area
                - Minimum engagement: Set thresholds for retweets, likes, and replies
                - News filter: Only show tweets that are news articles

                ## Pagination

                - Copy the pagination cursor that appears after a search
                - Paste it into the cursor field to load the next page of results

                ## Best Practices

                - Be specific with your search terms to get more relevant results
                - Use date ranges to narrow down results
                - Experiment with different result types (Top vs Latest)
                """
            )

st.markdown("---")
st.markdown("Developed with ❤️ by Chetanaya")
