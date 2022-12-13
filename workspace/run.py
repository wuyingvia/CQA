import os
import multiprocessing as mp


def run(model, dataset, method, cuda_idx):
    py = "/home/wuying/anaconda3/envs/torch2/bin/python"

    run_command = "{:s} -u {:s} --config ../configuration/{:s} --cuda {:d} --uncer_m {:s} --q {:f}".format(py, model, dataset, cuda_idx, method, q)
    print(run_command)
    os.system(run_command)


if __name__ == "__main__":

    
    #model_list = ['pred_STGCN12.py','pred_LSTNet12.py', 'pred_GraphWaveNet.py', 'pred_MTGNN.py', 'pred_GMAN.py']
    model_list = ['pred_GraphWaveNet.py']

    #dataset_list = ['PEMS03.conf', 'PEMS04.conf', 'PEMS08.conf', 'PEMSBAY.conf', 'PEMSD7M.conf', 'METR-LA.conf', 'PEMS07.conf',]
    dataset_list = ['PEMS03.conf', 'METR-LA.conf']

    method_list = ['quantile_conformal', 'quantile', 'adaptive']

    quantile_list = [0.5, 0.6, 0.7, 0.8, 0.95]


    limit = len(dataset_list)
    job_count = 0
    process_list = []

    for model in model_list:
        for dataset in dataset_list:
            for method in method_list:
                for q in quantile_list:

                    p = mp.Process(target=run, args=(model, dataset, method, job_count % 8))
                    p.start()
                    process_list.append(p)

                    if (job_count + 1) % limit == 0:
                        print(job_count)
                        for p in process_list:
                            p.join()

                    job_count = job_count + 1




# for model in model_list:

#     os.system('nohup python %s --config ../configuration/PEMS03.conf --cuda 0 --uncer_m quantile_conformal &'% (model))
#     os.system('nohup python %s --config ../configuration/PEMS03.conf --cuda 0 --uncer_m quantile_conformal &' % (model))