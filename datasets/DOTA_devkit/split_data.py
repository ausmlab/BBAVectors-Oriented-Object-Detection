from ImgSplit_multi_process import splitbase
src_path = '/BBAV/DS/train'
dst_path = r'examplesplit/train'

split = splitbase(src_path, dst_path, choosebestpoint=True, num_process=8)
print('splitting 0.5: ')
split.splitdata(0.5)
print('splitting 1: ')
split.splitdata(1)
print('splitting 2: ')
split.splitdata(2)

src_path = '/BBAV/DS/val'
dst_path = r'examplesplit/val'

split = splitbase(src_path, dst_path, choosebestpoint=True, num_process=8)
print('splitting 0.5: ')
split.splitdata(0.5)
print('splitting 1: ')
split.splitdata(1)
print('splitting 2: ')
split.splitdata(2)
