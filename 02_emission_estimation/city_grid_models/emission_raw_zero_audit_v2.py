"""Check prepared zeros against the original GeoTIFF, preserving masks."""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
ROOT=Path('${SERVER_HOME}/result')
OUT=ROOT/'emission_structural_optimization_v2'
d=pd.read_parquet(ROOT/'spatial_mobility_carbon_validation_v2/analysis/global_cell_features_v2.parquet')
parts=[]
for year in [2017,2024]:
    part=d[d.year.eq(year)].copy()
    with rasterio.open(ROOT/f'spatial_mobility_carbon_validation_v1/reference_tiff/{year}.tif') as src:
        raster=src.read(1,masked=True)
        idx=part.cell_id.to_numpy(int)
        assert raster.shape==(1800,3600)
        sample=raster.ravel()[idx]
        part['raw_tiff_nodata']=np.ma.getmaskarray(sample)
        part['raw_tiff_emission']=sample.filled(np.nan).astype(float)
        part['raw_tiff_transform']=str(src.transform)
    part['prepared_zero']=part.reference_emission.eq(0)
    part['raw_tiff_zero']=part.raw_tiff_emission.eq(0)
    part['raw_tiff_nonfinite']=~np.isfinite(part.raw_tiff_emission)
    parts.append(part[['cell_id','year','lon','lat','reference_emission','activity_total',
        'raw_tiff_nodata','raw_tiff_emission','prepared_zero','raw_tiff_zero','raw_tiff_nonfinite','raw_tiff_transform']])
out=pd.concat(parts,ignore_index=True)
out.to_csv(OUT/'raw_tiff_zero_audit.csv',index=False)
summary=out.groupby('year').agg(n=('cell_id','size'),prepared_zeros=('prepared_zero','sum'),
    raw_zeros=('raw_tiff_zero','sum'),raw_nodata=('raw_tiff_nodata','sum'),raw_nonfinite=('raw_tiff_nonfinite','sum'))
summary.to_csv(OUT/'raw_tiff_zero_summary.csv')
valid=~out.raw_tiff_nonfinite
assert np.allclose(out.loc[valid,'reference_emission'],out.loc[valid,'raw_tiff_emission'],rtol=1e-6,atol=1e-6)
print(summary.to_string(),flush=True)
