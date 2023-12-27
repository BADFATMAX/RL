import './App.css';
import Hello from './components/Hello';
import Message from './components/Message';
import Init from './pages/Init';
import Stop from './pages/Stop';
import Graph from './pages/Graph';
import Dataset from './pages/Dataset';
import NoPage from './pages/NoPage';

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
  var token = sessionStorage.getItem("token")
  console.log(token);
  if(!token) {
    return <Login setToken={setToken} />
  }
  return (
    <div className="App">
      {/* <p>Welcome! {token}</p> */}
      <BrowserRouter>
      <div className='sidebar'><Sidebar></Sidebar></div>
      <div className='pageContent'>
        <Routes>
          <Route index element={<Hello />} />
          <Route path="/init" element={ <Init/>}/>
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/result" element={<Graph />} />
          <Route path="/stop" element={<Stop />} />
          <Route path='*' element={<NoPage></NoPage>} />
        </Routes>
        </div>
      </BrowserRouter>
      {/* <Message MessageComponentContent="Called from App"/> */}
    </div>
  );
}

export default App;
