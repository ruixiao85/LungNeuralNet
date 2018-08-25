import os
import pandas as pd


def mk_dir_if_nonexist(_dir):
    if not os.path.exists(_dir):
        os.mkdir(_dir)

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


def find_file_recursive(_path, _ext):
    from glob import glob
    _images = [path for fn in os.walk(_path) for path in glob(os.path.join(fn[0], _ext))]
    _total = len(_images)
    print("Found [%d] file from [%s]" % (_total, _path))
    return _images
