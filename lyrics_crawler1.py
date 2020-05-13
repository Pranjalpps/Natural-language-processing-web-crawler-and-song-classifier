import re
import pathlib
import urllib.parse
import pathlib
import urllib.parse
from bs4 import BeautifulSoup
import requests
import os



def get_file_name(url):
    return pathlib.Path(urllib.parse.urlparse(url).path).name


def load_lyrics(name,primarylink):
    os.mkdir(name)
    os.chdir(name)
    resp = requests.get(primarylink)    
    soup = BeautifulSoup(resp.text, 'html.parser')
    lyrics=soup.find_all("a",attrs={"onmousedown":True})
    
    l1=[]
    for lyric in lyrics:
        p=str(lyric)
        if "Lyrics" not in p:
            continue
           
        l1.append(lyric.get('href'))
    l1=l1[5:105]     #initial links were useless so starting with 5
    
    TAG_RE = re.compile(r'<[^>]+>') #for removing tags
    for link in l1:
    
        resp1=requests.get(link)
        soup1=BeautifulSoup(resp1.text,'html.parser')
        unwanted=str(soup1.find("div",attrs={"class":"driver-related"}))
        text=str(soup1.find(attrs={"id":"lyrics-body-text"}))
        text=text.replace(unwanted," ")
        
        text=TAG_RE.sub('', text) #
        text = "".join([s for s in text.splitlines(True) if s.strip("\r\n")]) #removing empty lines
        fname = get_file_name(link).replace('.html','.txt')  #to save files as text files
        
        print (fname)
        with open(fname, "w") as f:
            f.write(text)
        f.close()    
    os.chdir('../')
    return

    
    
    
def main():
    
    dict={"mattalica":"https://www.metrolyrics.com/metallica-lyrics.html","snoop":"https://www.metrolyrics.com/snoop-dogg-lyrics.html",
          "bob":"https://www.metrolyrics.com/bob-dylan-lyrics.html","britney":"https://www.metrolyrics.com/britney-spears-lyrics.html"}

    for item in dict:
        load_lyrics(item,dict[item])
        
    
if __name__=="__main__":
    main()

