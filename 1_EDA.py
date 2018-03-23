
# coding: utf-8

# # You have some data... now what?
# 
# Usually, you want to spend some time **exploring the data**. This can include a bunch of things:
# - **Sanity checks**: does the data include the columns you're expecting? How are rows unique? Are there duplicates?
# - **Hunting for patterns and outliers**: Are there any? What do they look like? Does anything seem correlated?
# - **Some descriptive statistics** (mean, standard deviation, histograms)
# And just generally getting to know your data!
# 
# You can do a lot of this with `pandas`, but pictures contain $\infty$ words, so now's the time to introduce our visualization tools!
# 
# ## Next tool: `matplotlib`
# 
# ![matplotlib logo](imgs/matplotlib_logo.png)
# 
# [`matplotlib`](https://matplotlib.org/) is one of the most popular plotting libraries in the Python data science stack. You may find people using [`seaborn`](https://seaborn.pydata.org/) or [`bokeh`](https://bokeh.pydata.org/en/latest/), and definitely explore those and other options, but `matplotlib` is the "default" plotting library. It's very powerful and definitely worth investing time in.
# 
# However, `matplotlib`'s documentation is... not super friendly. (Unless you have experience [MATLAB](https://www.mathworks.com/products/matlab.html), in which case it may be familiar to you.) You can get a handle on it by keeping the following in mind:
# - The basic call for a `matplotlib` chart is `plt.plot(x, y)`.
#     - `.plot()`  has lots of additional, handy keyword arguments, like `color='red'`, `alpha=0.2` (20% transparency), and `label="Tuition"`.
# - `plt.plot()` doesn't always display the plot, though. Instead, it instantiates a matplotlib pyplot object.
# - You can explicitly display the object with `plt.show()`.
# - You can modify the object's attributes - e.g. the range of the x- and y-scales, the title, the x- and y-axis labels. For example, to change the title, you can do `plt.plot(x, y)`, then `plt.title("My cool chart title")` and then `plt.show()`!
# 
# Here's a good [walkthrough of the object-ness of `matplotlib` charts, and how to use it well](http://pbpython.com/effective-matplotlib.html), and [another one](https://realpython.com/blog/python/python-matplotlib-guide/). No need to go through these now, though! You can get pretty far with just the tips above!

# In[1]:


# Tell Jupyter to display matplotlib plots in your notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import matplotlib.pyplot as plt

# Let's resize all our charts to be a bit bigger
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 8)

# matplotlib has certain default styles that you can specify at the top of your notebook
# https://matplotlib.org/users/style_sheets.html
# Here, I use the 'bmh' style - just a personal preference!
plt.style.use('bmh')


# ### Back to the Greeks
# 
# Let's explore the Duke Greek life dataset.

# In[2]:


# Loading the Greek life data
df = pd.read_csv('data/duke-greek-life.csv')
print(df.shape)
df.head()


#  I don't really know anything about Duke or its Greek life, so I have a bunch of questions:
# - How many students are there? (I'm assuming the level of observation is a student.)
# - How many students are in a Greek organization?
# - What's the distribution of high school tuitions? (Do Duke students go to fancy high schools?)
# - How many had a merit scholarship?
# - Where are students from?

# In[3]:


# How many students are there?
len(df)


# In[4]:


# How many students are in a Greek organization?
df['Greek Organization'].value_counts()


# In[5]:


# How many Greek orgs are there?
df['Greek Organization'].nunique() - 1


# In[33]:


# What percentage of students are in a Greek org?
len(df[df['Greek Organization'] != "None"]) / len(df)


# In[6]:


# What's the distribution of high school tuitions?
df['Tuition of High School'].describe()


# Ah ha! Our first data cleaning
# 
# High school tuition sounds like it should be a **numeric** column, but the `dtype` above is `object`. (That is, `pandas` thinks that column is a string.) Why? 
# 
# Let's try to make it numeric with `pandas`'s method, `to_numeric()`:

# In[7]:


pd.to_numeric(df['Tuition of High School'])


# The comma in "16,600" is the problem. Let's try to correct it and see if that helps.

# In[8]:


df['Tuition of High School'].replace(to_replace="16,600", value="16600", inplace=True)
pd.to_numeric(df['Tuition of High School'])


# Hmmm. Still not finished. Also, it's probably not efficient to manually correct for every comma in every column. We can use a shortcut - Python's string methods! These are also available to us in `pandas`:

# In[9]:


pd.to_numeric(df['Tuition of High School'].str.replace(',',''))


# That worked! But we didn't actually change the column. Instead, `pandas` just returned a new Series for us (a Series is a column in a pandas DataFrame). Let's replace the column itself:

# In[10]:


df['Tuition of High School'] = pd.to_numeric(df['Tuition of High School'].str.replace(',',''))
df['Tuition of High School'].describe()


# Perfect! Now we can use one of `pandas`'s built-in plotting methods (yes, it has a few! they're built on top of `matplotlib`) to see a frequency histogram of this data.

# In[12]:


df['Tuition of High School'].hist()


# In[13]:


df['Tuition of High School'].isnull().value_counts()


# Let's modify this. Let's add:
# - More bins! I like more granular histograms.
# - A vertical line for the average tuition.

# In[14]:


df['Tuition of High School'].hist(bins=20)
plt.axvline(df['Tuition of High School'].mean(), 
            color='yellow', 
            label=f"Mean: ${df['Tuition of High School'].mean():,.2f}")
plt.legend()
plt.title('Distribution of high school tuitions\n(Duke University students)')
plt.xlabel('USD ($)')
plt.ylabel('Frequency')
plt.show()


# In[42]:


# Let's one hot encode (AKA make dummy variables) the public or private high school column
df['Public or Private High School'].value_counts()


# In[46]:


school_ohe = pd.get_dummies(df['Public or Private High School'], prefix='school_type')
school_ohe.head()


# In[50]:


# Let's add those dummies back into the original dataframe via merge!
df = pd.merge(df, school_ohe, left_index=True, right_index=True)
df.head()


# In[51]:


# I've forgotten - which columns do we have?
df.columns


# In[54]:


df['Greek Council'].value_counts() / len(df)


# In[73]:


# I don't really like typing out the long Proper English Words as column names,
# let's make them lowercase snake_case!
for col in df.columns:
    df.rename(columns={col: col.lower().replace(' ', '_') }, inplace=True)
    
# Check to make sure it worked    
df.head()


# ### One of the handiest `pandas` methods: `.apply()`
# 
# `pandas` has lots of built-in things you can do: summing across rows or columns, taking averages, changing the data types, and so on. But sometimes you want to do something specific and a bit more complicated. For example, you want to use multiple columns to calculate a value. In those cases, `.apply()` is helpful.
# 
# I highly recommend watching the [Data School](http://www.dataschool.io/) tutorial on `.apply()`:

# In[55]:


from IPython.display import YouTubeVideo
YouTubeVideo('P_q0tkYqvSk')


# I actually had trouble thinking of a good `.apply()` usage with this dataset! So here's a kinda crappy one: let's infer the biological sex of each student observation. We can do this by:
# 1. Assuming that `Fraternity == male` and `Sorority == female`.
# 2. Assuming that, if they aren't a member of a Greek organization, they have a 50% likelihood of being male/female.
# 
# This will be an imperfect measure, for a bunch of reasons. And we won't correctly assign sex to each observation, we'll just get the right "averages" out. 
# 
# Anyway, here's how `.apply()` works! (Side note: We're going to use [`scipy`](https://www.scipy.org/) - another **great data science tool** - to randomly sample from a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_trial).

# In[80]:


from scipy.stats import bernoulli

# This function will take each row as the input
def guessing_sex(row):
    """Male = 1, female = 0
    """
    if row['greek_council'] == 'Fraternity':
        return 1
    elif row['greek_council'] == 'Sorority':
        return 0
    else:
        return bernoulli.rvs(p=0.5)
    
# Applying that function to the dataframe
df['guessed_sex'] = df.apply(guessing_sex, axis=1)
df['guessed_sex'].value_counts() / len(df)


# In[84]:


# Sanity check
for membership in df['greek_council'].unique():
    print(f"\nNow checking: {membership}")
    print(df[df['greek_council']==membership]['guessed_sex'].value_counts() / len(df))


# # Onto [statistics and machine learning $\rightarrow$](2_StatsML.ipynb)
