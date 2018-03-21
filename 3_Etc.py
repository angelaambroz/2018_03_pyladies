
# coding: utf-8

# # More to learn 📚
# 
# Those are the core parts of the data science stack in Python:
# 
# - `pandas` (with the help of `sqlalchemy`, `requests`, or basic Python) to import and explore your data
# - `matplotlib` to visualize your data
# - `numpy`, `statsmodels` and/or `scikit-learn` to use some statistics or machine learning on your data
# 
# But then there's everything else! We won't have time for this, but there are a lot of other parts of the data stack - more "meta" parts. Knowing more about these makes you a stronger contributor and makes things, overall, more easy for you!
# 
# ## Extra power tools
# 
# ### Jupyter notebooks
# 
# One of the main tools we've been using all during this tutorial - and haven't mentioned once! - is [Jupyter](http://jupyter.org/)!
# 
# ![jupyter logo](imgs/jupyter_logo.png)
# 
# This thing here - what we're typing in and running - is a Jupyter notebook. Jupyter grew out of [IPython](https://ipython.org/), which was an interactive way to mix code with Markdown (simplified HTML) for easy-to-read and easy-to-share notebooks. 
# 
# Jupyter notebooks are where a lot of data science happens - especially the exploratory, visualization stuff. Right now, I'm using [Binder](https://mybinder.org/) to host these Jupyter notebooks online - and to have them be _interactive_! (That is, you can modify the code, run it, and see what happens!)
# 
# Using Jupyter is pretty straightforward. You just need to install it (e.g. `pip install jupyter`), launch it with `jupyter notebook` and then navigate - using your favorite web browser - to your `localhost`. 
# 
# ### Virtual environments
# 
# Often, code tutorials and workshops can get really painful because of _installation_. Everyone has a slightly different system, with a slightly different Python ecosystem on their machine, and so things can get messy and painful when trying to, say, get everyone to lauch their first Jupyter notebook on their machine! 
# 
# To avoid this problem, people often use **virtual environments**. Virtual environments are ways to keep a siloed, walled-off version of Python (and any packages you want to `pip` install) on your machine. You keep it distinct from your system Python (if you're using a Mac, for example, you have Python 2.7 installed already) and you keep your libraries from getting messy.
# 
# There are lots of ways to make a virtual environment, and I highly recommend [`pyenv`](https://github.com/pyenv/pyenv) combined with its add-on, [`pyenv virtualenv`](https://github.com/pyenv/pyenv-virtualenv). For example, for these workshop materials, I wanted to keep things clean and only include the bare minimum packages I would need to have everything run. To do that I did:
# 
# ```
# angela@home $ cd path/to/these/materials
# angela@home $ pyenv virtualenv pyladies
# angela@home $ pyenv local pyladies
# angela@home $ pip install jupyter pandas numpy scipy statsmodels scikit-learn matplotlib
# ```
# 
# This basically:
# 1. `pyenv virtualenv pyladies` $\rightarrow$ Created a `pyenv` virtual environment called `pyladies`.
# 2. `pyenv local pyladies` $\rightarrow$ Created a secret `.python-version` file, which tells `pyenv` to always use the `pyladies` virtual environment when I'm in this folder.
# 3. `pip install [blah blah]` $\rightarrow$ In this brand new Python environment I've made, install all the packages I'll need.
# 
# ### SQL and databases
# 
# SQL is its own beast, with its own syntax and all that. You can do a _lot_ of data exploration - and even some descriptive statistics - in SQL. I really recommend [Udacity's free course on relational databases](https://www.udacity.com/course/intro-to-relational-databases--ud197) to start your SQL journey.
# 
# # Recommended resources
# 
# #### Python
# - [Udacity: Programming Foundations with Python](https://www.udacity.com/course/programming-foundations-with-python--ud036)
# - many many many more...
# 
# #### Fun/interesting datasets
# - [Data is plural](https://tinyletter.com/data-is-plural/archive)
# 
# #### `pandas`
# - [Data School](https://www.youtube.com/user/dataschool)
# - [Chris Albon tutorials](https://chrisalbon.com/#Python)
# 
# #### SQL
# - [Udacity: Intro to relational databases](https://www.udacity.com/course/intro-to-relational-databases--ud197)
# 
# #### scikit-learn
# - [scikit-learn tutorials](http://scikit-learn.org/stable/tutorial/index.html)
# - YouTube!
# 
# #### statsmodels
# - [Docs](http://www.statsmodels.org/stable/index.html)
# - YouTube!
# 
# # A small plug
# 
# If you enjoyed this, I have more tutorials and stuff on data science, stats and Python at [my blog](http://www.angelaambroz.com/blog/).
# 
# # 🙏🙏🙏 Thanks! 🙏🙏🙏
