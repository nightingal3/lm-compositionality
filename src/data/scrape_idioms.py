from bs4 import BeautifulSoup
import requests 
import re
import pandas as pd
import pdb

url_1 = "https://www.ef.edu/english-resources/english-idioms/"
url_2 = "https://en.wikipedia.org/wiki/English-language_idioms"


def scrape_source_table(url) -> pd.DataFrame:
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "lxml")

    idioms = []
    meanings = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) <= 2:
            continue
        if "Further links" in tds[0].text:
            break
        idiom, meaning, _ = re.sub("[\r\n\t]+", "", tds[0].text), re.sub("[\r\n\t]+", "", tds[1].text), re.sub("[\r\n\t]+", "", tds[2].text)
        idioms.append(idiom.lower())
        meanings.append(meaning)

    return pd.DataFrame({"idiom": idioms, "meaning": meanings})
    
if __name__ == "__main__":
    df_1 = scrape_source_table(url_1)
    df_2 = scrape_source_table(url_2)
    print(df_1)
    print(df_2)
    combined = pd.concat([df_1, df_2])
    combined = combined.drop_duplicates(subset="idiom")
    combined.to_csv("./data/idioms/eng.csv", index=False)