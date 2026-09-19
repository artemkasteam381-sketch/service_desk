"""Small CPU integration benchmark, not a relevance test or service load test."""
from pathlib import Path
import argparse,json,time
import numpy as np
from .focused import load_source,BGEEncoder,AnalogSearch,METHODS
from .evaluation import MemoryMonitor


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    import torch
    torch.set_num_threads(2)
    queries=['Не приходит СМС с кодом входа в приложение','Не печатается этикетка для отправления',
        'После оплаты деньги списались, отправление не создалось','Не открывается личный кабинет, ошибка 500',
        'Не удается импортировать список отправлений из файла']
    with MemoryMonitor() as memory:
        _,records,texts,qvec,dvec,manifest,version=load_source(a.source)
        path=Path(a.source)/'model_cache/hub/models--BAAI--bge-m3/snapshots'/manifest['revision']
        t=time.perf_counter();encoder=BGEEncoder(path);load_ms=(time.perf_counter()-t)*1000
        # Checks that live query embeddings match the reused cache's model and pooling policy.
        i=next(i for i,t in enumerate(texts) if t)
        v=encoder.encode_queries([texts[i]])[0]
        cache_cosine=float(v@qvec[i])
        if cache_cosine<.999:raise ValueError(f'Live encoder does not match cache: cosine={cache_cosine}')
        engine=AnalogSearch(records,dvec,encoder)
        for m in METHODS:engine.search(queries[0],m)
        measurements=[]
        for q in queries:
            for m in METHODS:
                response=engine.search(q,m)
                measurements.append({'query':q,'method':m,**response['timing_ms']})
    report={'device':'cpu','torch_threads':2,'dataset_version':version,'model_load_ms':load_ms,
        'live_vs_cache_cosine':cache_cosine,'resources':memory.report,'measurements':measurements,
        'summary':{m:{'warm_p50_ms':float(np.median([x['total'] for x in measurements if x['method']==m])),
                      'warm_p95_ms':float(np.quantile([x['total'] for x in measurements if x['method']==m],.95))}
                   for m in METHODS},
        'limitations':'Only five short queries, sequential batch=1. Includes live query encoding. Not production p95 or human relevance evaluation. Shared model and index memory.'}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['summary','live_vs_cache_cosine','resources']},indent=2))


if __name__=='__main__':main()
