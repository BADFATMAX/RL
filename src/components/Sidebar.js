import { Link } from "react-router-dom";
import './Sidebar.css'
export default function Sidebar() {
    function doc(){
        console.log("asdasd");
        window.location.replace("http://192.168.253.179:3000");
      }
    return (
      <div className="sidebar">
      <Link to="/home"><button>Home</button></Link>
      {/* <br/>  */}
      <Link to="/init"><button>Init</button></Link>
      {/* <br/> */}
      <Link to="/result"><button>Results</button></Link>
      {/* <br/> */}
      <Link to="/api"><button>API</button></Link>
      {/* <br/>
      <Link to="/stop"><button>Stop</button></Link>
      <br/>*/}
      <button onClick={doc}>Doc</button>
      </div>
    );
  }