"""
Okay so this file is to get tweets from the database live in a file "twitDB2.vsc" on local path.
I'm trying to get some filteration on this file.
Now try to implement semantics to that file.
"""
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time

ckey = '53Oi8ofbfLqhANzJMYSd5O57O'
csecret = 'g8ZaVJ6ZU67uoMZjVvCjILfSLwzDvmlcX8vSAQGynCEI1bJAKk'
atoken = '142544162-StI793M4NP8SrZSfMrM2hoiWhzPh5h9YSoXGwhoO'
asecret = 'sUY3YRLBTWh877tuFjeDfTrbOcfbTDkbE0ZcAPBJKJZ66'

class listener(StreamListener):

    def on_data(self, data):
        try:
            tweet = data.split(',"text":"')[1].split('","source')[0]
            print tweet
            saveFile = open('twitDB2.csv','a')
            saveThis = str(time.time())+':::'+tweet
            saveFile.write(saveThis)
            saveFile.write('\n')
            saveFile.close()
            return True
        except BaseException, e:
            print 'failed ondata,', str(e)
            time.sleep(5)
    def on_errror(self, status):
        print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["superman"])
