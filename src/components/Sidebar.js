import { Link } from "react-router-dom";
import './Sidebar.css'
export default function Sidebar() {
    return (
      <div className="sidebar">
      <Link to="/init"><button>Init</button></Link>
      <br/>
      <Link to="/result"><button>Graph</button></Link>
      {/* <br/>
      <Link to="/stop"><button>Stop</button></Link>
      <br/>
      <Link to="/dataset"><button>Dataset</button></Link> */}
      </div>
    );
  }