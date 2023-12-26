from typing import Union

from fastapi import FastAPI, BackgroundTasks
import subprocess as sp
import os
import sys
import signal
# import tempfile

app = FastAPI()

app.running_nns = {}

def id2fld(id):
    return f"{id}_nn_task"

def run_example(id: str):
    fld = id2fld(id)
    try:
        _ = app.running_nns[id]
        print(f"{id} is already running")
    except KeyError:
        fp = f"{fld}/out.tmp"
        with open(fp, "w") as fptr:
            fptr = open(fp, "w")
            p = sp.Popen([sys.executable, f"{fld}/learn.py"], stdout=fptr)
            app.running_nns.update({id: {"fptr": fptr, "sp": p}})
            print(f"{id} is started")
            p.wait()
            fptr.close()
        del app.running_nns[id]
        os.unlink(fp)   


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ex_run")
def example_run(background_tasks: BackgroundTasks):
    id = "ex"
    fld = id2fld(id)
    if not os.path.exists(fld):
        print("no such task!")
        return {"None"}
    background_tasks.add_task(run_example, id)
    return {"nn_id": id}

@app.get("/ex_init")
def example_init():
    id = "ex"
    fld = id2fld(id)
    if os.path.exists(fld):
        print("task is already exists")
        return {"None"}
    p = sp.Popen([sys.executable, "example_init.py"])
    p.wait()
    print("task is inited")
    return {"OK"}

@app.get("init/{nn_task}")
def init(nn_task: str, dset: str = "custom", los:str = "custom", opt:str = "custom"):
    id = nn_task
    fld = id2fld(id)
    if os.path.exists(fld):
        print("task is already exists")
        return {"None"}
    

@app.get("/res/{nn_task}")
def get_result(nn_task: Union[str, None] = None):
    print(app.running_nns)
    if nn_task is None:
        return {"None"}
    else:
        try:
            nn =  app.running_nns[nn_task]
            net = None
            with open(nn["fptr"].name, "r") as fptr:
                content = fptr.read()
                try:
                    net = content.split("RENDER!")[-1].split("/RENDER")[0]
                except Exception:
                    pass
                fptr.close()
            return {"content": net}
        except KeyError:
            return {"None"}

@app.get("/del/{nn_task}")
def finish_nn(nn_task: Union[str, None] = None):
    print(app.running_nns)
    if nn_task is None:
        return {"None"}
    else:
        try:
            nn =  app.running_nns[nn_task]
            nn["sp"].kill()
            return {"OK"}
            
        except KeyError:
            return {"None"}




@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}