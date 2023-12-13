import './App.css';
import Hello from './components/Hello';
import Message from './components/Message';
import Start from './pages/Start';
import Stop from './pages/Stop';
import Result from './pages/Result';
import Dataset from './pages/Dataset';
import NoPage from './pages/NoPage';

import { BrowserRouter, Routes, Route} from 'react-router-dom';

import { Link } from "react-router-dom";

function App() {

  return (
    <div className="App">
      <BrowserRouter>
      <Link to="/result"><button>Go to result</button></Link>
      <Link to="/start"><button>Go to start</button></Link>
      <Link to="/stop"><button>Go to stop</button></Link>
      <Link to="/dataset"><button>Go to dataset</button></Link>
        <Routes>
          <Route index element={<Hello />} />
          <Route path="/start" element={<Start />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/result" element={<Result />} />
          <Route path="/stop" element={<Stop />} />
          <Route path='*' element={<NoPage></NoPage>} />
        </Routes>
      </BrowserRouter>
      <Message MessageComponentContent="Called from App"/>
    </div>
  );
}

export default App;
