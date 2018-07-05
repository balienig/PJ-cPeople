import os
pathFloderStore = 'train/StoreFace/'

list = []




for folder , dirs, files in os.walk(pathFloderStore):
    # print(folder)
    nameFolder = folder.split('/')
    if nameFolder[2] != "" :
        list.append(nameFolder[2])
    # print(nameFolder[2])
    # for file in files:
    #     path = os.path.join(folder,file)
    #     word_label = path.split('/')
    #     print(word_label[2])

print(list)