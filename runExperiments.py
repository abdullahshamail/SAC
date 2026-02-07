from utils import pretty_print_convoys, make_synthetic_like_real, calibrate_density
import time
import pandas as pd

from brute_force import BruteForce
from ed_sac import ED_SAC
from ed_sac_grid import ED_SAC_Grid


def makeSynethticData():
    print("Calibrating and generating synthetic data...")
    hint = calibrate_density(N=2000, T=50, eps=0.02, target_median=36.0, target_p95=290.0)
    params = dict(
            N=2000, T=50, D=2,
            eps=0.02, m=10, k=5,
            num_convoys=250, convoy_size_range=(10, 250),
            life_range=(10, 50),
            center_step=hint['center_step'],
            jitter_frac=hint['jitter_frac'],
            background_frac=0.35,
            background_drift=0.003,

            semantics_mode="sbert",   
            semantic_purity=0.1,      
            purity_on=(True, True, True)
            )
    X, userToEmb, features, meta, keyToAttr = make_synthetic_like_real(**params)
    

    return X, userToEmb, features, keyToAttr, params['T'], params['N'], params['eps'], params['k'], params['m']

def main():

    X, userToEmb, features, keyToAttr, Tframes, Nobj, epsilon, tempK, tempM = makeSynethticData()
    dthreshold = 0.3
    print(f"Running experiments with eps={epsilon}, k={tempK}, m={tempM}, d_div={dthreshold}...")
    alg3 = ED_SAC_Grid(eps=epsilon, k=tempK, m=tempM, d_thresh=dthreshold, features=features, userToEmb=userToEmb, keyToAttr=keyToAttr, attr_names=("demographics","sports","color"))

    t0 = time.perf_counter(); conv3 = alg3.fit_predict(X); t1 = time.perf_counter()

    numConv3, timeTaken3 = pretty_print_convoys(conv3, title="ED-SAC-Grid", timeTaken=f"{t1-t0:.2f}s", T = Tframes, n=Nobj, eps=epsilon, div_val=dthreshold, k = tempK)


    alg2 = ED_SAC(eps=epsilon, k=tempK, m=tempM, dThreshold=dthreshold, features=features, userToEmb=userToEmb, keyToAttr=keyToAttr, attr_names=("demographics","sports","color"))

    t0 = time.perf_counter(); conv2 = alg2.fit_predict(X); t1 = time.perf_counter()

    numConv2, timeTaken2 = pretty_print_convoys(conv2, title="Ed-SAC", timeTaken=f"{t1-t0:.2f}s", T = Tframes, n=Nobj, eps=epsilon, div_val=dthreshold, k = tempK)


    alg1 = BruteForce(eps=epsilon, k=tempK, m=tempM, dThreshold=dthreshold, features=features, userToEmb=userToEmb, keyToAttr=keyToAttr)

    t0 = time.perf_counter(); conv1 = alg1.fit_predict(X); t1 = time.perf_counter()

    numConv1, timeTaken1 = pretty_print_convoys(conv1, title="Brute-Force", timeTaken=f"{t1-t0:.2f}s", T = Tframes, n=Nobj, eps=epsilon, div_val=dthreshold, k = tempK)

    data_rows = []
    data_rows.append({
    'epsilon': epsilon,
    'k': tempK,
    'd_div': dthreshold,
    'time_BF': timeTaken1,
    'time_ED_SAC': timeTaken2,
    'time_ED_SAC_GRID': timeTaken3,
    'num_SACs_BF': numConv1,
    'num_SACs_ED_SAC': numConv2,
    'num_SACs_ED_SAC_GRID': numConv3
})
    runTimes = pd.DataFrame(data_rows)

    print("=*50", runTimes)


if __name__ == "__main__":
    main()