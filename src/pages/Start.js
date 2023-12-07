/* import Hello from '../components/Hello';

export default function Start() {
    return (
        <>
            <Hello />
            <h2>Start Page</h2>
        </>
    )
} */

import Hello from '../components/Hello';

import { useNavigate } from 'react-router-dom';

export default function Start() {
  const navigate = useNavigate();

  const handleButtonClick = () => {
    navigate('/dataset');
  };

  return (
    <>
      <Hello />
      <h2>Start Page</h2>
      <button onClick={handleButtonClick}>Go to Dataset</button>
    </>
  );
}

