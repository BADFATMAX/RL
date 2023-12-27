import { useState } from "react"
import "./Graph.css"
async function getGraph(){
  let ret = "";
    if (sessionStorage.getItem("INITED")){
      try {
        let r = await fetch("http://127.0.0.1:8000/res/ex");
        let raw = await r.json();
        console.log(raw);
        ret = raw["content"]
        // ret = <div>{String(raw["content"])}</div>;
      }
      catch (er){
        alert(er);
      }
    }
    return ret;
}

function getBtn(){
  async function handler(){
    console.log("handler");
    try{
      let resp = await fetch("http://127.0.0.1:8000/ex_download");
      let blob = await resp.blob();
      let file = window.URL.createObjectURL(blob);
      window.location.assign(file);
      console.log(resp);
    }
    catch (er){
      alert(er);
    }
  }
  let ret = <></>
  if (sessionStorage.getItem("INITED")){
    return <div><button onClick={handler}>Download</button></div>;
  }
}
export default function Graph() {
  const [graph, setGraph] = useState();
  // const [downloadBt, setdwnBt] = useState();
  getGraph().then((res) => setGraph(res));
  // setdwnBt(getBtn());
  return (
    <div className="Graph">
      <h1>Visualization and results</h1>
      {/* <button onClick={handleButtonClick}>Go to Stop</button> */}
      {graph}
      {getBtn()}
    </div>
  );
}
