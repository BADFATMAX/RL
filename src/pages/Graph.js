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
export default function Graph() {
  const [graph, setGraph] = useState();
  getGraph().then((res) => setGraph(res));
  return (
    <div className="Graph">
      <h1>Visualization and results</h1>
      {/* <button onClick={handleButtonClick}>Go to Stop</button> */}
      {graph}
    </div>
  );
}
