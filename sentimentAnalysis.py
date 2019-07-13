# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:19:21 2018

@author: Ankit_Kumar34
"""
import tweepy
from textblob import textblob

consumer_key='GxRIdJFIjF07oCX3piE9isXiQ'
consumer_secret='gegoikIVZpQzfrsUgSSUsIegeIPrkIfaNgKMwWuJxOCy7hjznx'

access_token='83552483-HMISvRE87HVbjpLOlTKOmJEqs0BTNlfC7sMBRYiAR'
access_token_secret='TPyeTL2o1TGE3mffc539CWq399kDM03ZdY4o1ghatnsy6'
auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth)
public_tweets=api.search('Trump')
for tweet in public_tweets:
    print(tweet.text)
    analysis=TextBlob(tweet.text)
    print(analysis.sentiment)