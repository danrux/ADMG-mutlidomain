"""One small generation -> fitting -> graph-evaluation example."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
METHOD_DGPS={'MuDo-nll':'scaling','SP':'mask','UMNI':'hard_intervention',
             'diagGMM':'auxillary','VaDE-linear':'auxillary','LiNGAM':'scaling',
             'BANG':'scaling','DCD':'scaling','RCD':'scaling'}

def synthetic(method,output):
    command=[sys.executable,'-B',str(ROOT/'main.py'),'--dgp',METHOD_DGPS[method],
             '--baseline',method,'--n','3','--T','2','--N','200','--seeds','2',
             '--graph-dense','1.0','--confounder-dense','0.5','--num-steps','30',
             '--num-initializations','1','--batch-size','200','--log-every','30',
             '--scheduler','none','--patience','0','--umni-workers','1',
             '--dcd-num-restarts','1','--dcd-max-iterations','2',
             '--dcd-optimizer-max-iterations','100','--rcd-max-samples','60',
             '--model-dir',str(output)]
    subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONUTF8':'1',
                   'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLBACKEND':'Agg'},check=True)
    records=[]
    for path in output.rglob('metrics.json'):
        record=json.loads(path.read_text(encoding='utf-8'))
        if record.get('method')==method and record.get('status')=='complete':
            records.append(record)
    if len(records)!=1:
        raise RuntimeError('Expected exactly one completed example run')
    return records[0]['metrics']

def causalassembly(method,output):
    sys.path.insert(0,str(ROOT))
    import numpy as np
    import torch
    from experiments.causalassembly_admg.config import ExperimentConfig,InterventionConfig,MethodConfig
    from experiments.causalassembly_admg.graph import load_causalassembly_graph,select_graph_split,edge_matrices
    from experiments.causalassembly_admg.interventions import build_intervention_design
    from experiments.causalassembly_admg.linear import generate_linear_dataset
    from experiments.causalassembly_admg.methods import fit_and_evaluate_method
    from experiments.result_store import write_json
    torch.set_num_threads(1)
    config=ExperimentConfig(setup='linear',n_regimes=4,n_per_regime=120,
        n_repetitions=1,bidirected_min_abs_correlation=0.10,
        interventions=InterventionConfig(n_targets=5,max_targets_per_regime=2),
        methods={method:MethodConfig(num_steps=20,lr=0.001 if method=='diagGMM' else 0.01,
                                    batch_size=256,scheduler='none',patience=0,log_every=20)})
    context=load_causalassembly_graph()
    split=select_graph_split(context,config.split)
    design=build_intervention_design(context.station12_graph,split,config.interventions,config.n_regimes)
    dataset=generate_linear_dataset(context.station12_graph,split,design,config,config.base_seed)
    metrics,arrays=fit_and_evaluate_method(method,dataset,split,config,config.base_seed)
    output.mkdir(parents=True,exist_ok=True)
    directed,bidirected=edge_matrices(split)
    np.savez_compressed(output/'graphs.npz',method=np.asarray(method),
        directed_true=directed,bidirected_true=bidirected,**arrays)
    np.savez_compressed(output/'dataset.npz',observed_nodes=np.asarray(dataset.observed_nodes),
        **{f'train_{index:02d}':domain for index,domain in enumerate(dataset.train_domains)})
    write_json(output/'metrics.json',{'method':method,'dgp':'causalassembly-linear',
               'seed':config.base_seed,'status':'complete','metrics':metrics})
    write_json(output/'run_config.json',config.as_dict())
    return metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=list(METHOD_DGPS),default='MuDo-nll')
    parser.add_argument('--dataset',choices=['synthetic','causalassembly'],default='synthetic')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.dataset=='causalassembly' and args.method not in {'MuDo-nll','diagGMM'}:
        parser.error('causalAssembly supports MuDo-nll and diagGMM in this example')
    output=(args.output or ROOT/'outputs/quickstart'/args.dataset/args.method).resolve()
    metrics=synthetic(args.method,output) if args.dataset=='synthetic' else causalassembly(args.method,output)
    print('Directed F1:',metrics['directed']['f1'])
    print('Bidirected F1:',metrics['bidirected']['f1'])
    print('Completed example:',output)

if __name__=='__main__':
    main()
