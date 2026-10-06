"""Write per-run metrics and graph arrays without campaign infrastructure."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np

def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key):json_ready(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [json_ready(item) for item in value]
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    return str(value)

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(json_ready(value),indent=2,sort_keys=True)+'\n',encoding='utf-8')
    temporary.replace(path)

class RunResultStore:
    def __init__(self,args: Any,*,T: int,x_n: int):
        self.args=args
        self.T=int(T)
        self.x_n=int(x_n)

    def artifact_directory(self,save_dir,seed: int) -> Path:
        directory=Path(save_dir)
        if self.args.evaluate:
            directory=directory/'evaluation'
        directory.mkdir(parents=True,exist_ok=True)
        return directory

    def write_metrics(self,*,seed:int,save_dir,metrics,diagnostics) -> Path:
        path=self.artifact_directory(save_dir,seed)/'metrics.json'
        write_json(path,{'method':self.args.method,'dgp':self.args.dgp,'seed':int(seed),
                         'status':'evaluated' if self.args.evaluate else 'complete',
                         'metrics':metrics,'diagnostics':diagnostics})
        return path

    def write_array_artifact(self,*,seed:int,save_dir,filename:str,values) -> Path:
        if Path(filename).name!=filename or not filename.endswith('.npz'):
            raise ValueError('filename must be a plain .npz filename')
        path=self.artifact_directory(save_dir,seed)/filename
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle,**{key:value for key,value in values.items() if value is not None})
        temporary.replace(path)
        return path

    def record_admg(self,*,seed:int,save_dir,num_domains:int,truth,estimate,
                    metrics,runtime_seconds:float,mixing=None) -> None:
        self.write_array_artifact(seed=seed,save_dir=save_dir,filename='graphs.npz',values={
            'graph_kind':np.asarray('admg'),'method':np.asarray(self.args.method),
            'seed':np.asarray(seed),'directed_true':truth.directed,
            'directed_hat':estimate.directed,'bidirected_true':truth.bidirected,
            'bidirected_hat':estimate.bidirected,'directed_weights_true':truth.directed_weights,
            'directed_weights_hat':estimate.directed_weights,'A_hat_raw':mixing,
            'runtime_seconds':np.asarray(runtime_seconds)})
        configuration={key:value for key,value in vars(self.args).items() if key!='seeds'}
        configuration.update(seed=int(seed),effective_T=self.T,observed_dimension=self.x_n,
                             num_domains=int(num_domains))
        write_json(self.artifact_directory(save_dir,seed)/'run_config.json',configuration)
