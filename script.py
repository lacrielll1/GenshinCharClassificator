from py3pin.Pinterest import Pinterest
import requests

def search_name_in_url(url):
    return url.split('/')[-1]

pinterest = Pinterest(email='Здесь введи email на Pinterest', password='Здесь пароль', username='Здесь username')
pinterest.login()

chars = ['Albedo', 'Amber', 'Ayaka', 'Ayato', 'Barbara', 'Beidou', 'Bennet', 'Chongyun', 'Collei', 'Diluc',
 'Diona', 'Eula', 'Fischl', 'Ganyu', 'Gorou', 'Heizou', 'Hu Tao', 'Itto', 'Jean', 'Kaeya', 'Kazuha', 'Keqing', 'Klee',
 'Kokomi', 'Lisa', 'Mona', 'Ningguang', 'Noelle', 'Qiqi', 'Raiden Shogun', 'Razor', 'Rosaria', 'Sara', 'Sayu',
 'Shenhe', 'Shinobu', 'Sucrose', 'Tartaglia', 'Thoma', 'Tighnari', 'Venti', 'Xiangling', 'Xiao', 'Xingqiu', 'Xinyan', 'Yae Miko',
 'Yanfei', 'Yelan', 'Yoimiya', 'Yun Jin', 'Zhongli']

for char in chars:
    path = 'g:/proj_data/train/' + char + '/'
    results = []
    search_batch = pinterest.search(scope='pins', query=char + ' genshin impact')
    for batch in search_batch:
        results.append(batch['images']['orig']['url'])
    for url in results:
        name = search_name_in_url(url)
        extension = name.split('.')[1]
        if extension == 'gif':
            continue
        img_data = requests.get(url).content
        with open(path + name, 'wb') as handler:
            handler.write(img_data)