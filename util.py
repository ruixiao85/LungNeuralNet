import os


def get_recursive_rel_path(_wd, _sf, ext='*.jpg'):
    _path = os.path.join(_wd, _sf)
    from glob import glob
    images = [path for fn in os.walk(_path) for path in glob(os.path.join(fn[0], ext))]
    total = len(images)
    print("Found [%d] file from subfolders [/%s] of [%s]" % (total, _sf, _wd))
    for i in range(total):
        images[i] = os.path.relpath(images[i], _path)
    return images, total


def mk_dir_if_nonexist(_dir):
    if not os.path.exists(_dir):
        os.mkdir(_dir)
