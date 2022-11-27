import os
import multiprocessing as mp


def run(model, dataset, method, cuda_idx):
    py = "/home/wuying/anaconda3/envs/torch2/bin/python"

    run_command = "{:s} -u {:s} --config ../configuration/{:s} --cuda {:d} --uncer_m {:s}".format(py, model, dataset, cuda_idx, method)
    print(run_command)
    os.system(run_command)


if __name__ == "__main__":

    
    #model_list = ['pred_STGCN12.py','pred_LSTNet12.py', 'pred_GraphWaveNet.py', 'pred_MTGNN.py', 'pred_GMAN.py']
    model_list = ['pred_GMAN.py']
    dataset_list = ['PEMS03.conf', 'PEMS04.conf', 'PEMS08.conf', 'PEMSBAY.conf', 'PEMSD7M.conf', 'METR-LA.conf', 'PEMS07.conf',]
    method_list = ['adaptive']

    limit = len(dataset_list)
    job_count = 0
    process_list = []

    for model in model_list:
        for dataset in dataset_list:
            for method in method_list:

                p = mp.Process(target=run, args=(model, dataset, method, job_count % 4))
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