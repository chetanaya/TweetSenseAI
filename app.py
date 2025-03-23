import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import os
from dotenv import load_dotenv
import yaml

load_dotenv()

# Load configuration from config.yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract configuration values
filters_config = config.get("filters", {})
dynamic_filters = filters_config.get("dynamic_filters", [])
default_max_tweets = filters_config.get("default_max_tweets", 40)

# Set page title and favicon
st.set_page_config(page_title="Twitter Search Dashboard", page_icon="🐦", layout="wide")

# Initialize session state variables
if "next_cursor" not in st.session_state:
    st.session_state.next_cursor = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "tweets_df" not in st.session_state:
    st.session_state.tweets_df = None
if "has_searched" not in st.session_state:
    st.session_state.has_searched = False

# Dashboard title
st.title("🐦 Twitter Search Dashboard")
st.markdown("Search for tweets using Twitter's Advanced Search API")

# Sidebar for API credentials and search parameters
with st.sidebar:
    st.header("API Credentials")
    api_key = st.text_input(
        "API Key", type="password", value=os.getenv("TWITTER_API_KEY", "")
    )

    st.header("Search Parameters")
    # Basic query and date range
    query = st.text_input("Search query", placeholder="Enter keywords, hashtags, etc.")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End date", datetime.now())

    st.subheader("Additional Filters")

    # Tweets to fetch
    max_tweets = st.number_input(
        "Tweets to Fetch",
        min_value=20,
        max_value=500,
        value=default_max_tweets,
        step=20,
        help="Number of tweets to fetch (will make multiple API calls if needed)",
    )

    # Container for dynamic filters (rendered based on config)
    st.markdown("#### Dynamic Filters")
    dynamic_filter_values = {}
    for filter_item in dynamic_filters:
        if not filter_item.get("enabled", True):
            continue
        key = filter_item.get("key")
        label = filter_item.get("label", key)
        ftype = filter_item.get("type", "text")
        default = filter_item.get("default", "")
        # Render input field based on its type
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
        else:  # Fallback: text input
            value = st.text_input(label, value=default, key=key)
        dynamic_filter_values[key] = value

    # Additional query type selection (e.g., Latest or Top)
    query_type_options = ["Latest", "Top"]
    query_type = st.selectbox("Query Type", options=query_type_options)

    # Pagination cursor (if needed)
    cursor = st.session_state.next_cursor

    # Function to search tweets with a given cursor
    def search_tweets_with_cursor(current_cursor=""):
        base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Build the base query parts
        query_parts = [query, f"since:{start_date_str}", f"until:{end_date_str}"]

        language_mapping = filters_config.get("language_mapping", {})

        # Loop over dynamic filters and add to query if their value differs from default
        for filter_item in dynamic_filters:
            if not filter_item.get("enabled", True):
                continue
            key = filter_item.get("key")
            ftype = filter_item.get("type", "text")
            default = filter_item.get("default", "")
            query_prefix = filter_item.get("query_prefix", "")
            value = dynamic_filter_values.get(key)

            # Special handling for language filter using the mapping from config
            if key == "language":
                if value != "All":
                    mapped_value = language_mapping.get(value, value)
                    query_parts.append(f"{query_prefix}{mapped_value}")
                continue  # Skip generic processing for language

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
                query_parts.append(f"{query_prefix}{value}")

        final_query = " ".join(query_parts)
        payload = {
            "query": final_query,
            "queryType": query_type,
            "cursor": current_cursor,
        }
        # print(payload)  # For debugging purposes; remove in production
        headers = {"X-API-Key": api_key}
        try:
            response = requests.get(base_url, headers=headers, params=payload)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

    # Function to fetch multiple pages of tweets (pagination)
    def search_tweets():
        all_tweets = []
        all_results = {"tweets": []}
        current_cursor = cursor
        total_tweets_fetched = 0
        page_count = 0

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

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

        progress_bar.empty()
        status_text.empty()
        all_results["tweets"] = all_tweets
        return all_results

    # Buttons to search and clear results
    search_button_sidebar = st.button("Search Tweets", key="search_btn_sidebar")
    clear_results = st.button("Clear Results")
    if clear_results:
        st.session_state.search_results = None
        st.session_state.tweets_df = None
        st.session_state.has_searched = False
        st.session_state.next_cursor = ""
        st.rerun()

# Main content area
if not api_key:
    st.warning("Please enter your API key in the sidebar to start searching.")
    st.info("You can obtain an API key from https://docs.twitterapi.io")
    st.subheader("Sample Dashboard Preview")
    st.image(
        "https://placehold.co/800x400?text=Twitter+Dashboard+Preview",
        use_column_width=True,
    )
else:
    if search_button_sidebar:
        if not query:
            st.warning("Please enter a search query.")
        else:
            # Reset the pagination cursor for a fresh search
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
                st.rerun()

    if (
        st.session_state.has_searched
        and st.session_state.search_results
        and st.session_state.tweets_df is not None
    ):
        results = st.session_state.search_results
        tweets_df = st.session_state.tweets_df
        st.success(f"Found {len(tweets_df)} tweets matching your search criteria.")

        if not tweets_df.empty:
            st.subheader("Tweet Metrics Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_likes = (
                    tweets_df["likeCount"].mean()
                    if "likeCount" in tweets_df.columns
                    else 0
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
                tweet_counts = (
                    tweets_df.groupby("date").size().reset_index(name="count")
                )
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
                    "created_at": tweets_df["created_at"]
                    if "created_at" in tweets_df.columns
                    else pd.to_datetime(tweets_df["createdAt"]),
                    "likes": tweets_df["likeCount"]
                    if "likeCount" in tweets_df.columns
                    else pd.Series(0, index=tweets_df.index),
                    "retweets": tweets_df["retweetCount"]
                    if "retweetCount" in tweets_df.columns
                    else pd.Series(0, index=tweets_df.index),
                    "replies": tweets_df["replyCount"]
                    if "replyCount" in tweets_df.columns
                    else pd.Series(0, index=tweets_df.index),
                    "quotes": tweets_df["quoteCount"]
                    if "quoteCount" in tweets_df.columns
                    else pd.Series(0, index=tweets_df.index),
                    "views": tweets_df["viewCount"]
                    if "viewCount" in tweets_df.columns
                    else pd.Series(0, index=tweets_df.index),
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
            if len(engagement_df) > 0:
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

            st.subheader("Tweet Results")
            tweet_filter = st.text_input(
                "Filter tweets",
                placeholder="Type to filter results",
                key="tweet_filter",
            )
            filtered_tweets = tweets_df
            if tweet_filter and "text" in tweets_df.columns:
                filtered_tweets = tweets_df[
                    tweets_df["text"].str.contains(tweet_filter, case=False, na=False)
                ]
            tweet_display = st.radio(
                "Display Style",
                ["Card View", "Table View", "JSON View"],
                key="display_style",
            )
            if tweet_display == "Card View":
                for _, tweet in filtered_tweets.iterrows():
                    try:
                        text_preview = (
                            tweet.get("text", "")[:100] + "..."
                            if "text" in tweet and len(tweet["text"]) > 100
                            else tweet.get("text", "No text")
                        )
                        with st.expander(text_preview):
                            try:
                                author = tweet.get("author", {})
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
                                created_at = tweet.get("createdAt", "")
                                st.markdown(
                                    f"**{author_name}** (@{author_username}) - {created_at}"
                                )
                                if "text" in tweet:
                                    st.markdown(tweet["text"])
                                else:
                                    st.warning("No text content available")
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric("❤️ Likes", tweet.get("likeCount", 0))
                                with col2:
                                    st.metric(
                                        "🔄 Retweets", tweet.get("retweetCount", 0)
                                    )
                                with col3:
                                    st.metric("💬 Replies", tweet.get("replyCount", 0))
                                with col4:
                                    st.metric("🔁 Quotes", tweet.get("quoteCount", 0))
                                with col5:
                                    st.metric("👁️ Views", tweet.get("viewCount", 0))
                                st.markdown(
                                    f"**Source:** {tweet.get('source', 'Unknown')}"
                                )
                                st.markdown(
                                    f"**Language:** {tweet.get('lang', 'Unknown')}"
                                )
                                if tweet.get("isReply", False):
                                    st.markdown(
                                        f"**Reply to:** @{tweet.get('inReplyToUsername', 'Unknown')}"
                                    )
                                if "entities" in tweet and isinstance(
                                    tweet["entities"], dict
                                ):
                                    entities = tweet["entities"]
                                    if "hashtags" in entities and entities["hashtags"]:
                                        hashtags = [
                                            f"#{tag['text']}"
                                            for tag in entities["hashtags"]
                                            if isinstance(tag, dict) and "text" in tag
                                        ]
                                        if hashtags:
                                            st.markdown(
                                                f"**Hashtags:** {', '.join(hashtags)}"
                                            )
                                    if (
                                        "user_mentions" in entities
                                        and entities["user_mentions"]
                                    ):
                                        mentions = [
                                            f"@{mention['screen_name']}"
                                            for mention in entities["user_mentions"]
                                            if isinstance(mention, dict)
                                            and "screen_name" in mention
                                        ]
                                        if mentions:
                                            st.markdown(
                                                f"**Mentions:** {', '.join(mentions)}"
                                            )
                                    if "urls" in entities and entities["urls"]:
                                        urls = [
                                            f"[{url.get('display_url', url.get('url', ''))}]({url.get('expanded_url', url.get('url', ''))})"
                                            for url in entities["urls"]
                                            if isinstance(url, dict)
                                        ]
                                        if urls:
                                            st.markdown(f"**URLs:** {' | '.join(urls)}")
                                tweet_id = tweet.get("id", "")
                                if tweet_id:
                                    tweet_url = tweet.get(
                                        "url",
                                        f"https://twitter.com/twitter/status/{tweet_id}",
                                    )
                                    st.markdown(f"[View on Twitter]({tweet_url})")
                                else:
                                    st.warning("No tweet ID available for link")
                            except Exception as e:
                                st.error(f"Error displaying tweet details: {str(e)}")
                                st.json(tweet)
                    except Exception as e:
                        st.error(f"Error processing tweet: {str(e)}")
            elif tweet_display == "Table View":
                try:
                    display_columns = []
                    if "author" in tweets_df.columns:
                        display_columns.append("Author")
                    if "text" in tweets_df.columns:
                        display_columns.append("Text")
                    if "createdAt" in tweets_df.columns:
                        display_columns.append("Created At")
                    metrics_columns = []
                    for col in [
                        "likeCount",
                        "retweetCount",
                        "replyCount",
                        "quoteCount",
                        "viewCount",
                    ]:
                        if col in tweets_df.columns:
                            metrics_columns.append(col)
                    table_data = {}
                    if "author" in tweets_df.columns:
                        table_data["Author"] = filtered_tweets["author"].apply(
                            lambda x: (
                                f"@{x.get('userName', '')}"
                                if isinstance(x, dict) and "userName" in x
                                else "Unknown"
                            )
                        )
                    if "text" in tweets_df.columns:
                        table_data["Text"] = filtered_tweets["text"]
                    if "createdAt" in tweets_df.columns:
                        table_data["Created At"] = filtered_tweets["createdAt"]
                    for col in metrics_columns:
                        display_name = col.replace("Count", "")
                        display_name = display_name[0].upper() + display_name[1:]
                        table_data[display_name] = filtered_tweets[col]
                    if table_data:
                        table_df = pd.DataFrame(table_data)
                        st.dataframe(table_df, use_container_width=True)
                    else:
                        st.warning("No data available for table view")
                except Exception as e:
                    st.error(f"Error creating table view: {str(e)}")
                    st.dataframe(filtered_tweets, use_container_width=True)
            else:  # JSON View
                st.json(filtered_tweets.to_dict(orient="records"))

            st.subheader("Export Results")
            col1, col2 = st.columns(2)
            with col1:
                csv = tweets_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"twitter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv",
                )
            with col2:
                json_str = tweets_df.to_json(orient="records")
                st.download_button(
                    "Download JSON",
                    data=json_str,
                    file_name=f"twitter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json",
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
