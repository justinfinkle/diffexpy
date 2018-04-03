import pandas as pd
from pydiffexp.gnw.results import GnwNetResults

def cc(pp=False):
    gnr = GnwNetResults(data_dir, experimental='ko')
    t = [0, 15, 40, 90, 180, 300]
    gnr.compile_results(censor_times=t, save_intermediates=False, pp=pp)

if __name__ == '__main__':
    pd.set_option('display.width', 2000)
    data_dir = "../../data/motif_library/gnw_networks/"

    # Make a list of all the subdirectories
    gnr = GnwNetResults(data_dir, experimental='ko')
    t = [0, 15, 30, 60, 120, 240, 480]
    # tt = timeit.timeit('cc(True)', number=4, globals=globals())
    # print(tt)
    all_stats = gnr.compile_results(censor_times=t)
    print(all_stats.shape)
    all_stats.to_csv('../intermediate_data/ko-wt_sim_stats_strongly_connected_2_censoredtimes.tsv', sep='\t')