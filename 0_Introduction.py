
# coding: utf-8

# # Introduction
# 
# So you want to science some data. Where to start? 
# 
# The typical steps of a data science workflow are:
# ![process](imgs/process.png)
# -- source: [CS109B - Advanced Topics in Data Science, Harvard](https://canvas.harvard.edu/courses/20101/assignments/syllabus)
# 
# Let's **skip** the first step ("Ask an interesting question"), and focus on the rest. How do we get data into Python? How do we explore it? What packages are available for statistics and machine learning? What about visualization?
# 
# # Step 1: Getting data
# 
# These days, data shows up in a variety of formats:
# - Files on disk (e.g. `some_data.csv`, `my_excel_stuff.xlsx`, `some_nested_stuff.json`, `some_picture.png`)
# - Structured data in "the cloud" (an API, such as [The New York Times's](https://developer.nytimes.com/))
# - Databases (e.g. SQL and NoSQL)
# - Streaming (e.g. Apache Kafka)
# 
# > In order of frequency, the most common sources of data for me have been:
# - Databases (MySQL, PostgreSQL, Vertica)
# - `.csv` files
# - APIs
# - Miscellaneous riff-raff
# 
# <hr>
# 
# # First (and most important) tool
# ![pandas-logo](imgs/pandas_logo.png)
# 
# [`pandas`](https://pandas.pydata.org/), developed by Wes McKinney in the late 2000s, is the Swiss Army knife of data science in Python. With `pandas`, you can:
# - Import data from (almost) any source (files, APIs, databases, tables on websites).
# - Convert typical Python data structures (e.g. `dict`, `list`) into `pandas` dataframes.
# - Visualize data.
# - Create/calculate/drop columns, take means and standard deviations, and generally clean and explore your data!
# 
# What you _can't_ do with `pandas`:
# - Fit a statistical or machine learning model.
# 
# Let's use `pandas` to grab data from some different formats!
# 
# > **Pro tip**
# 
# > `pandas` defines **dataframes** as 2-dimensional arrays of data, i.e. a tabular data format. This is similar to `R`.
# 
# > Importing data into pandas (always) follows the same pattern: `pd.read_*(file_source)`. For example, `pd.read_csv('some_csv.csv')`, `pd.read_sql('select * from some_table', sql_connection_object)`
# 
# From the [docs](https://pandas.pydata.org/pandas-docs/stable/io.html):
# ![pandas-io](imgs/pandas_io.png)
# 
# ### In action: Let's import a `.csv` of Duke University's Greek life
# 
# Data source: [github.com/Crissymbeck](https://github.com/Chrissymbeck/Greek-Life-Demographics) Found via: [Data is Plural newsletter - 24 Jan 2018 edition](https://tinyletter.com/data-is-plural/letters/data-is-plural-2018-01-24-edition)

# In[1]:


import pandas as pd

df = pd.read_csv('data/duke-greek-life.csv')
print(df.shape)
df.head()


# If your data is coming from an API or a SQL database, you need a couple more tools to get it into pandas.
# 
# ## Second tool: SQL and `sqlalchemy`
# 
# With SQL, you need:
# 1. a database! 😊
# 2. a query in some dialect of SQL (e.g. `SELECT * FROM SOME_TABLE`)
# 3. a way to connect Python to the database!
# 
# We won't cover (1) and (2) today (though I'll include some learning materials in TODO), but there are lots of ways to do (3). I use the [`sqlalchemy`](https://www.sqlalchemy.org/) library. But there are others ([lots](https://pypi.python.org/pypi?:action=search&term=sql)!), and many offer the same basic functionality - tomayto, tomahto.
# 
# #### Local vs. remote database
# 
# For the purposes of this workshop, I took the Duke Greek life dataset `.csv` and created a small, local [`sqlite`](https://www.sqlite.org/index.html) database (`sqlite` is just one of many flavors of SQL databases - it's handy when you want a small database that you save on your local computer). Normally, SQL databases in the "real/working world" will be hosted on some server (i.e. some other computer somewhere) and you'll need credentials (like a username and password) to access it. Apart from adding in those credentials into your "connection string" (`sqlite:///data/greek.db`), the process is basically the same.

# In[6]:


from sqlalchemy import create_engine

sql_connection = engine = create_engine('sqlite:///data/greek.db')
sql_df = pd.read_sql('select * from duke', sql_connection)
sql_df.head()


# ### Third tool: APIs and `requests` and Star Wars, oh my
# 
# Sometimes you want to grab data from the, _ahem_, "cloud". That is, the data isn't hosted on a database that you have access to, nor is it on any files that you can download. Nowadays, many organizations offer their data via an API. But what is an API?
# 
# An API is an [Application Programming Interface](https://en.wikipedia.org/wiki/Application_programming_interface). It's a structured way to share data across the web. Many websites will have (1) a user-facing front end and (2) an API. For example:
# 
# - [The New York Times](https://www.nytimes.com/) and [The New York Times API](https://developer.nytimes.com/)
# - [Weather Underground](https://www.wunderground.com/) and the [Weather Underground API](https://www.wunderground.com/weather/api/).
# - [and many more!](https://www.programmableweb.com/apis)
# 
# Every API is a bit different. Meaning: read their docs! Some APIs are better documented than others. But the basic process is always the same: You access the data in an API by making an **HTTP request** to a specific **endpoint** (i.e. a URL). That endpoint normally sends the data back in `.json` format (though it might send it back as an `.xml`, or something else!).  Often you'll need to register your intent to use an API, and you'll be given an **API key** - this is a way for the API maintainers to "rate limit" users. That is, to prevent users from bombarding their API with millions of requests per day.
# 
# Since grabbing data from an API necessitates (1) talking over HTTP and (2) dealing with the `json` data format, we'll need a couple more tools: the `json` library (which comes with the Python standard library) and [`requests`](http://docs.python-requests.org/en/master/) (which you need to install on your machine with `pip install requests`).
# 
# Let's use a simple API - one that doesn't require an API key: the [Star Wars API](https://swapi.co/)!
# 
# #### First, we use the `requests` library to issue an HTTP `GET` request from the Star Wars API's `people` endpoint

# In[8]:


import json
import requests

swapi_url = 'https://swapi.co/api/people/'
r = requests.get(swapi_url)
r.status_code


# #### Success! HTTP responses always come with [status codes](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes) - "200" means "all OK!"
# 
# Now let's convert the HTTP response's "payload" (i.e. the data it sent back to us, which will be a string that can be json-ified) into `json` and then, finally, into a `pandas` dataframe.

# In[9]:


# Raw text of the Star Wars API's response (only the first 100 characters)
r.text[0:200]


# In[11]:


# The requests library has a handy method called .json() which converts an HTTP response to json for you!
r.json()


# Looking at the above `json`, we see that the `count` of data points is 87, the `next` page is at a new endpoint (`https://swapi.co/api/people/?page=2`) and, finally, the actual data we're interested in is in the `results` key. 
# 
# Remember: every API is different, and their responses will be slightly differently structured! It's common to see JSON, but not always guaranteed. It's also common to see a response that includes some metadata, like the `count` of data observations or other things, and then a `results` key that includes the actual data. It also sometimes happen that, rather than returning _all_ the data points (in this case, _all_ the characters in Star Wars), it returns only some amount and then divides the rest on additional `pages` (like `?page=2` above). 
# 
# Let's take a look at the `results`! `pandas` is very handy - you can just feed it a `.json()` and tell it you want to convert it into a dataframe!

# In[12]:


# We just want the actual people, which can be found in the JSON's 'results' key
starwars_df = pd.DataFrame(r.json()['results'])
starwars_df.head()


# In[13]:


# Let's find my favorite character to confirm everything is working OK
starwars_df[starwars_df['name']=='Obi-Wan Kenobi']


# In[14]:


# What?! We know Obi-Wan's homeworld?
starwars_df[starwars_df['name']=='Obi-Wan Kenobi']['homeworld']


# Obi-Wan's homeworld is a place called [_Stewjon_](http://starwars.wikia.com/wiki/Stewjon)?! Named after Jon Stewart?! 
# 
# ## So that's how you grab data
# 
# There's a _lot_ more to, for example, SQL querying or parsing HTTP responses, but those are the basics of how to get things into a `pandas` dataframe. Now, let's take a look at [cleaning and exploring the data](1_EDA.ipynb)!
# 
# # Next: [Exploring data $\rightarrow$](1_EDA.ipynb)
