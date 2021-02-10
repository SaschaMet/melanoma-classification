import pandas as pd


def read_shapefile(sf):
    # fetching the headings from the shape file
    fields = [x[0] for x in sf.fields][1:]
    # fetching the records from the shape file
    records = [list(i) for i in sf.records()]
    shps = [s.points for s in sf.shapes()]
    # converting shapefile data into pandas dataframe
    df = pd.DataFrame(columns=fields, data=records)
    # assigning the coordinates
    df = df.assign(coords=shps)
    return df
