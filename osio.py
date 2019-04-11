import os
import pandas as pd
from glob import glob


def mkdir_ifexist(_dir):
    exist=os.path.exists(_dir)
    if not exist: os.mkdir(_dir)
    return exist
def mkdir_dir(_dir):
    exist=os.path.exists(_dir)
    if not exist: os.mkdir(_dir)
    return _dir
def mkdirs_dir(_dir):
    _path=os.path.dirname(_dir)
    os.makedirs(_path,exist_ok=True)
    return _dir


def to_excel_sheet(_df, _xls, _sheet):
    if os.path.exists(_xls):
        from openpyxl import load_workbook
        book = load_workbook(_xls)
        writer = pd.ExcelWriter(_xls, engine='openpyxl')
        writer.book = book
        _df.to_excel(writer, sheet_name=_sheet)
        writer.save()
        writer.close()
    else:
        _df.to_excel(_xls, sheet_name=_sheet)

def find_folder_prefix(_path, _prefix):
    _items=[_dir for _dir in os.listdir(_path) if os.path.isdir(os.path.join(_path,_dir)) and _dir.startswith(_prefix)]
    _total=len(_items)
    print("Found [%d] folders start with [%s] from [%s]" % (_total, _prefix,  _path))
    return _items

def find_file_pattern(_path, _pattern):
    _items=glob(os.path.join(_path,_pattern))
    _total=len(_items)
    print("Found [%d] files matching [%s] from [%s]" % (_total, _pattern,  _path))
    return _items

def find_file_pattern_rel(_path, _pattern):
    return [os.path.relpath(abspath,_path) for abspath in find_file_pattern(_path, _pattern)]

def find_file_ext_recursive(_path, _ext):
    _images = [path for fn in os.walk(_path) for path in glob(os.path.join(fn[0], _ext))]
    _total = len(_images)
    print("Found [%d] files recursively matching [%s] from [%s]" % (_total, _ext, _path))
    return _images

def find_file_ext_recursive_rel(_path, _ext):
    return [os.path.relpath(abspath,_path) for abspath in find_file_ext_recursive(_path, _ext)]

