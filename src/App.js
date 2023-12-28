import './App.css';
import Hello from './components/Hello';
import Message from './components/Message';
import Init from './pages/Init';
import Stop from './pages/Stop';
import Graph from './pages/Graph';
import Home from './pages/Home';
import NoPage from './pages/NoPage';
import API from './pages/API';

import { BrowserRouter, Routes, Route} from 'react-router-dom';

import { Link } from "react-router-dom";
import Sidebar from './components/Sidebar';

import React, {useState} from 'react';

import Login from './components/Login/Login';

// function defineToken(){
//   let token = sessionStorage.getItem("token")
//   if (!token){

//   }
// }

function App() {
  const [tokenToSet, setToken] = useState();
  function doc(){
    console.log("asdasd");
    window.location.replace("http://10.9.54.200:3000");
  }
  var token = sessionStorage.getItem("token")
  console.log(token);
  if(!token) {
    return <Login setToken={setToken} />
  }
  return (
    <div className="App">
      {/* <p>Welcome! {token}</p> */}
      {/* <button onClick={doc}>DOC</button> */}
      <BrowserRouter>
      <div className='sidebar'><Sidebar></Sidebar></div>
      <div className='pageContent'>
        <Routes>
          <Route index element={<Home />} />
          <Route path="/init" element={ <Init/>}/>
          <Route path="/home" element={<Home />} />
          <Route path="/result" element={<Graph />} />
          <Route path="/stop" element={<Stop />} />
          <Route path='/api' element= {<API/>}/>
          <Route path='*' element={<NoPage></NoPage>} />
        </Routes>
        </div>
      </BrowserRouter>
      {/* <Message MessageComponentContent="Called from App"/> */}
    </div>
  );
}

export default App;
