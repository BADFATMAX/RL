import { useState } from 'react';
import './Init.css'
import Select from 'react-dropdown-select';

async function getPosDatasets(){
  let options = JSON.parse(sessionStorage.getItem("possibleDatasets"));
  console.log("from storage: ", options);
  if (!options){
    options = []
    try{
      let response = await fetch("http://127.0.0.1:8000/datasets");
      let raw = await response.json();
      options = [];
      for (let i = 1; i <= raw.length; ++i){
        options.push({id: i, name: raw[i-1]});
      }
      console.log("response", raw);
      console.log("options", JSON.stringify(options));
      sessionStorage.setItem("possibleDatasets", JSON.stringify(options));
    } catch(er){
      alert(er);
    }
  }
  return options;
}

async function runNN(){
  try{
    let response = await fetch("http://127.0.0.1:8000/ex_run");
    // console.log("runNN");
  }
  catch(er){
    alert(er);
  }
}

async function stopNN(){
  try{
    let response = await fetch("http://127.0.0.1:8000/stop/ex");
    // console.log("runNN");
  }
  catch(er){
    alert(er);
  }
}

export default function Init() {
  const [val, setDataset] = useState();
  const [options, setOptions] = useState([]);
  const [iterations, setIter] = useState();
  const [resp, setResp] = useState(null);

  async function handleSubmit(e){
    e.preventDefault();
    if (val && iterations){
      let checkStat = sessionStorage.getItem("INITED");
      if (checkStat == null){
        sessionStorage.setItem("dataset", JSON.stringify(val[0]));
        sessionStorage.setItem("iterations", iterations);
        sessionStorage.setItem("INITED", true);
      }
      let response = await fetch("http://127.0.0.1:8000/ex_init");
      let raw = await response.json();
      setResp(raw[0]);
    }
    console.log(sessionStorage.getItem("dataset"), sessionStorage.getItem("iterations"), resp);
  }

  function initInfo(){
    let iters = sessionStorage.getItem("iterations");
    let dataset = JSON.parse(sessionStorage.getItem("dataset"))
    if (dataset)
      dataset = dataset["name"];
    // let status = sessionStorage.getItem("INITRESP");
    let header = <h2>RL initialized with {iters} number of iterations and {dataset} dataset</h2>;
    let message = <></>;
    if (resp == "None")
      message = <p>The neural network has already been initialized</p>;
    let runb = <button onClick={runNN}>Run!</button>
    let stopb = <button onClick={stopNN}>Stop!</button>
    if (iters && dataset){
      return <>{message}{header}{runb}{stopb}</>;
    }
    else
      return <></>;
  }

  return (
    <div className='Init'>
      <h1>Init your training</h1>
      {/* <button onClick={handleButtonClick}>Go to Dataset</button> */}
      <h2>Please specify on which dataset the RL neural network should be trained and the maximum number of iterations</h2>
      <form onSubmit={handleSubmit}>
        <label>
          <p>Number of iterations</p>
          <input type='number' onChange={e => setIter(Number(e.target.value))}/>
        </label>
        <Select onDropdownOpen={() => {getPosDatasets().then((opts) => setOptions(opts));}} options={options} labelField="name" valueField="id" onChange={(values) => setDataset(values)} />
        <div>
        <button type="submit">Submit</button>
      </div>
      </form>
      <div>{initInfo()}</div>
    </div>
  );
}

