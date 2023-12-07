import Hello from '../components/Hello';
/*
export default function Stop() {
    return (
        <>
            <Hello />
            <h2>Success Page</h2>
        </>
    )
} */

import { useNavigate } from 'react-router-dom';

export default function Stop() {
  const navigate = useNavigate();

  const handleTrainMoreClick = () => {
    navigate('/result');
  };

  const handleDatasetClick = () => {
    navigate('/dataset');
  };

  const handleDownloadClick = () => {
    // Add your download logic here
  };

  return (
    <>
      <Hello />
      <h2>Success Page</h2>
      <button onClick={handleTrainMoreClick}>Train more</button>
      <button onClick={handleDatasetClick}>Go to Dataset</button>
      <button onClick={handleDownloadClick}>Download</button>
    </>
  );
}
