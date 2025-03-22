# TweetSenseAI

An LLM-powered application for in-depth analysis of Twitter data based on keyword searches.

## Overview

TweetSenseAI is a tool that allows users to search for tweets using Twitter's Advanced Search API, visualize the results, and analyze engagement metrics. The application features an intuitive user interface built with Streamlit that makes it easy to search, filter, and export Twitter data.

## Features

- **Advanced Search Queries**: Construct complex Twitter searches using keywords, hashtags, Boolean operators, and more
- **Customizable Filters**: Filter tweets by language, location, engagement metrics, and date range
- **Automatic Pagination**: Fetch multiple pages of tweets automatically up to a specified limit
- **Data Visualization**: View tweet volume over time and engagement metrics in interactive charts
- **Multiple Display Options**: View tweets in card format, table format, or raw JSON
- **Export Capabilities**: Download tweet data in CSV or JSON format for further analysis
- **Session State Management**: Results persist when interacting with UI elements

## Installation

### Prerequisites
- Python 3.7+
- Twitter API credentials (from [twitterapi.io](https://docs.twitterapi.io))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/chetanaya/TweetSenseAI.git
cd TweetSenseAI
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Twitter API key:
```
TWITTER_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application in your web browser at `http://localhost:8501`

3. Enter your API key in the sidebar (if not provided in the `.env` file)

4. Configure your search parameters:
   - Enter search query (keywords, hashtags, mentions, etc.)
   - Set date range
   - Choose language
   - Configure additional filters
   - Set maximum number of tweets to fetch

5. Click "Search Tweets" to execute the search

6. View and interact with the results:
   - Toggle between Card View, Table View, and JSON View
   - Filter displayed tweets
   - View engagement metrics and charts
   - Export data to CSV or JSON

## API Key

The application requires an API key from Twitter's API provider. You can obtain an API key by:

1. Visiting [twitterapi.io](https://docs.twitterapi.io)
2. Signing up for an account
3. Creating a new project to get your API key

## Query Syntax

The search query supports Twitter's advanced search syntax, including:

- Exact phrases: `"climate change"`
- Boolean operators: `climate AND action`
- Exclusions: `-fossil`
- Hashtags: `#climateaction`
- Mentions: `@username`
- Combined operators: `climate (action OR policy) -"climate change denial"`
- Source filters: `from:username`
- Mention filters: `to:username`
- Reply filters: `conversation_id:1234567890`

## Future Plans

This project is currently in development. Future enhancements will include:

- LLM-powered sentiment analysis of tweets
- Topic modeling and content categorization
- User network analysis
- Geographical visualization of tweet origins
- Scheduled searches and alerts
- Customizable dashboards and reports
- Integration with other NLP tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed by Chetanaya
- Uses Twitter's Advanced Search API for data retrieval