import { useState } from "react"
import "./Graph.css"
import axios from "axios";
// import { Buffer } from "buffer";
// async function getGraph(){
//   let ret = "";
//     if (sessionStorage.getItem("INITED")){
//       try {
//         let r = await fetch("http://127.0.0.1:8000/res/ex");
//         let raw = await r.json();
//         console.log(raw);
//         ret = raw["content"]
//         // ret = <div>{String(raw["content"])}</div>;
//       }
//       catch (er){
//         alert(er);
//       }
//     }
//     return ret;
// }

async function getFile(){
  let ret = null;
    if (sessionStorage.getItem("INITED")){
      try {
        let r = await axios.get("http://127.0.0.1:8000/res/ex",  { responseType: 'blob' });
        let raw = await r.data;
        // let raw = Buffer.from(r.data, 'binary').toString('base64');
        // var base64String = btoa(String.fromCharCode.apply(null, new Uint8Array(raw)));
        // console.log("img", r.data);
        ret = raw;
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
  if (sessionStorage.getItem("INITED") == true || sessionStorage.getItem("INITED") == 'true'){
    return <div><button className="btn1" onClick={handler}>Download</button></div>;
  }
}

export default function Graph() {
  // const [graph, setGraph] = useState();
  const [file, setFile] = useState(null);
  function drawFile(){
    console.log("drawing file:", file);
    if (file != null) {

      console.log("file: ", typeof(file));
      // return <p>sdfsdf</p>
      return (<div><br/><img src={URL.createObjectURL(file)}/></div>);
    }
    else
      return <></>
  }
  function reloadImage(){
    getFile().then((res) => setFile(res))
  }

  // const [downloadBt, setdwnBt] = useState();
  // getGraph().then((res) => setGraph(res));
  // setdwnBt(getBtn());
  return (
    <div className="Graph">
      <h1>Visualization and results</h1>
      {/* <button onClick={handleButtonClick}>Go to Stop</button> */}
      {/* {graph} */}
      <button className="btn1" onClick={reloadImage}>Update</button>
      {drawFile()}
      {getBtn()}
    </div>
  );
}
