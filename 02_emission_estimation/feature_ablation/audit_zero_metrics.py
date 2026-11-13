import pandas as pd
import numpy as np

path = r'${TRANSPORT_WORKSPACE}\\Emission_Estimation_Validation_Study_V3\Experiments\grid_random\42\test_predictions.csv'
df = pd.read_csv(path)
print(df.regime.value_counts().to_string())
for regime, data in df.groupby('regime'):
    for label, subset in [('all', data), ('positive', data[data.reference_raw > 0])]:
        observed = subset.observed_log
        predicted = subset.Validation_blend
        r2 = 1 - np.sum((observed - predicted) ** 2) / np.sum((observed - observed.mean()) ** 2)
        print(
            regime, label, 'n=', len(subset), 'zeros=', int((subset.reference_raw == 0).sum()),
            'r2=', r2,
            'mae=', np.mean(np.abs(observed - predicted)),
            'rmse=', np.mean((observed - predicted) ** 2) ** 0.5,
        )
    zeros = data[data.reference_raw == 0]
    print(regime, 'mean predicted log at reference-zero cells=', zeros.Validation_blend.mean())
