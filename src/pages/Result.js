import Hello from '../components/Hello';
/*
export default function Result() {
    return (
        <>
            <Hello />
            <h2>Graph Page</h2>
        </>
    )
}*/

import { useNavigate } from 'react-router-dom';

export default function Result() {
  const navigate = useNavigate();

  const handleButtonClick = () => {
    navigate('/stop');
  };

  return (
    <>
      <Hello />
      <h2>Graph Page</h2>
      <button onClick={handleButtonClick}>Go to Stop</button>
    </>
  );
}
