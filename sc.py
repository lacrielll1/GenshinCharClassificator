from py3pin.Pinterest import Pinterest
import requests

def search_name_in_url(url):
    return url.split('/')[-1]

def download_board(val_board_data, path):
    id = val_board_data[4]['id']
    pins = pinterest.board_feed(board_id=id)
    p = path + val_board_data[0] + '/'
    for pin in pins:
        if 'images' not in pin.keys():
            continue
        url = pin['images'][list(pin['images'].keys())[-1]]['url']
        name = search_name_in_url(url)
        extension = name.split('.')[1]
        if extension == 'gif':
            continue
        img_data = requests.get(url).content
        with open(p + name, 'wb') as handler:
            handler.write(img_data)

pinterest = Pinterest(email='Здесь введи email на Pinterest', password='Здесь пароль', username='Здесь username')
pinterest.login()
ctr = 0
path = 'g:/proj_data/train/'
with open('boards.txt') as f:
    valid = []
    
    while (True):
        line = f.readline()
        req = line.split(',')
        if len(req) < 3:
            break
        char = req[0]
        username = req[1]
        board = req[2].replace('\n', '')
        boards = pinterest.boards(username=username)
    
        for b in boards:
            if b['name'] == board:
                valid.append([char, username, board, b['pin_count'], b])

d = dict()
for b_info in valid:
    if b_info[0] in d.keys():
        d[b_info[0]] += int(b_info[3])
    else:
        d[b_info[0]] = int(b_info[3])
print(d)

for val in valid:
    print(val[0])
    download_board(val, path)
