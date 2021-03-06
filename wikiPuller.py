from requests import get
from re import findall
from json import dumps
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
class Grapher:
    def __init__(self,start_url):
        self.visited = {}
        self.connections = {}
        self.finished = {}
        self.edges = 0
        self.scraper(start_url)

    def find_branches(self,raw_data):
        whitelist = []
        raw_data = raw_data.split('<a href="/wiki/')[1:]
        blacklist = ['Wikipedia:','Wikipedia_talk','File:','Special:','User:','Category:','Portal:','Help:',"Main_Page",'Template:','Geo_(','User_talk','Talk:']
        for attempt in raw_data:
            #print(f"\n\n\n\n\n\n{attempt}-> {list(filter(lambda x : attempt.find(x) > -1, blacklist))}")
            splitted = attempt.split('" title=')
            if (len(splitted) > 1) and not list(filter(lambda x : attempt.find(x) > -1, blacklist)):
                disambig_end = splitted[0].find("_(disambig")
                redir_end = splitted[0].find('" class="mw-redirect')
                if disambig_end > 0:
                    whitelist.append(splitted[0][:disambig_end])
                elif redir_end > 0:
                    whitelist.append(splitted[0][:redir_end])
                else:
                    whitelist.append(splitted[0])
        return whitelist

    def scrape(self,base_url):
        self.finished[base_url] = True
        print(f"total: {len(self.finished)} : found: {base_url}")
        print(f"nodes: {len(self.connections)}\nedges: {self.edges}")
        finds = []
        raw_data = get(f"https://en.wikipedia.org/wiki/{base_url}",verify=False,timeout=4).content.decode()
        branches = self.find_branches(raw_data)
        self.connections[base_url] = {}
        for name in branches:
            self.edges += 1
            self.connections[base_url][name] = True
            self.visited[name] = True
        return

    def scraper(self,url):
        self.visited[url] = True
        iter = 0
        while self.visited and self.edges < 1000000:
            iter += 1
            operating_node = list(self.visited.keys())[0]
            try:
                self.scrape(operating_node)
            except:
                try:
                    self.scrape(operating_node)
                except:
                    try:
                        self.scrape(operating_node)
                    except:
                        del self.visited[operating_node]

            del self.visited[operating_node]

if __name__ == "__main__":
    g = Grapher('Ba_Congress')
    graph = open("wikiGraph",'w')
    graph.write(dumps(g.connections))
