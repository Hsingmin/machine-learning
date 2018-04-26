#coding:utf-8

import ocr.keys as keys

extern = 'extern massage...'

def show_keys():
    print('__ import keys in same package __')
    characters = keys.alphabet[:]
    print(characters[0:100])
    print(extern)



