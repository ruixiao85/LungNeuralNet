import argparse
import pandas as pd

from osio import to_excel_sheet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process pandas excel files.')
    parser.add_argument('-f', '--file', dest='file', action='store',
                        default='Original1536x1536_2019-07.22 Kyle Monica BKB1 PiZ Lung Morph_MRCNN_Vgg16_TanhReluSoft3Dca.xlsx', help='both headered/indexed pandas excel file.')
    parser.add_argument('-s', '--sheet', dest='sheet', action='store',
                        default='', help='sheet name if available.')
    parser.add_argument('-r', '--region', dest='region', action='store',
                        default='RespiratoryAirway', help='region to subset, e.g., RespiratoryAirway')
    parser.add_argument('-c', '--category', dest='category', action='store',
                        default="", help='category to subset, e.g., MONO')
    parser.add_argument('-p', '--parameter', dest='parameter', action='store',
                        default='CountDensity', help='parameter to subset, e.g., CountDensity')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='output.xls', help='output file after processing')
    args = parser.parse_args()
    sheet_name=None if len(args.sheet)<1 else args.sheet
    df = pd.read_excel(args.file,header=[0,1],index_col=[0,1],sheet_name=sheet_name) if sheet_name else \
        pd.read_excel(args.file,header=[0,1],index_col=[0,1])
    idx=pd.IndexSlice
    regs=[x for x in args.region.split(',') if x]
    cats=[x for x in args.category.split(',') if x]
    pars=[x for x in args.parameter.split(',') if x]
    if cats:
        dfs=df.loc[idx[:,regs],idx[cats,pars]]
    else:
        dfs=df.loc[idx[:,regs],idx[:,pars]]
    # print(dfs)
    print(f"Processing from DataFrame of shape {df.shape} to {dfs.shape}.")
    to_excel_sheet(dfs,args.output,sheet_name or "0")